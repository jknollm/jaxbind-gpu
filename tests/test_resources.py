import gc
import weakref
import threading
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


def test_context_is_only_available_in_external_callbacks():
    with pytest.raises(RuntimeError, match="callback"):
        jb.current_context()


def test_stream_wrapper_reused_without_reusing_borrowed_arrays(gpu):
    streams = []
    arrays = []

    def external(x, *, out):
        streams.append(jb.current_context().stream)
        arrays.append((weakref.ref(x), weakref.ref(out)))
        gpu.add(x, 1, out=out)

    compiled = jax.jit(jb.bind(external, out=True))
    inputs = [jnp.full(16, i, dtype=jnp.float32) for i in range(4)]
    outputs = [compiled(x) for x in inputs]
    jax.block_until_ready(outputs)
    for i, output in enumerate(outputs):
        np.testing.assert_array_equal(output, np.full(16, i + 1))
    assert len({id(stream) for stream in streams}) == 1
    jb.synchronize()
    gc.collect()
    assert all(reference() is None for pair in arrays for reference in pair)


def test_external_library_thread_local_state_survives_callbacks(gpu):
    local = threading.local()
    handles = {}

    def external(x, *, out):
        if not hasattr(local, "handle"):
            local.handle = object()
        handles.setdefault(threading.get_ident(), []).append(local.handle)
        gpu.add(x, 1, out=out)

    compiled = jax.jit(jb.bind(external, out=True))
    x = jnp.arange(16, dtype=jnp.float32)
    for _ in range(8):
        np.testing.assert_array_equal(compiled(x), np.arange(16) + 1)
    assert any(len(values) > 1 for values in handles.values())
    for values in handles.values():
        assert all(value is values[0] for value in values)


def test_executable_owns_callback_and_releases_it_after_destruction(gpu):
    def make():
        weights = gpu.full(1024, 2, dtype="float32")
        reference = weakref.ref(weights)

        def external(out, args, config):
            gpu.multiply(args[0], weights, out=out[0])

        callback_reference = weakref.ref(external)
        op = jb.get_call(external, lambda x: ((x.shape, x.dtype),))
        compiled = (
            jax.jit(op).lower(jax.ShapeDtypeStruct((1024,), np.float32)).compile()
        )
        return compiled, reference, callback_reference

    compiled, reference, callback_reference = make()
    gc.collect()
    assert callback_reference() is not None
    np.testing.assert_array_equal(
        compiled(jnp.ones(1024, dtype=jnp.float32))[0], np.full(1024, 2)
    )
    del compiled
    jax.clear_caches()
    jb.synchronize(clear_cache=True)
    gc.collect()
    assert callback_reference() is None
    assert reference() is None


def test_lowered_program_owns_callback_before_compilation(gpu):
    def make():
        def external(out, args, config):
            gpu.add(args[0], 3, out=out[0])

        reference = weakref.ref(external)
        op = jb.get_call(external, lambda x: ((x.shape, x.dtype),))
        return jax.jit(op).lower(jax.ShapeDtypeStruct((4,), np.float32)), reference

    lowered, reference = make()
    gc.collect()
    assert reference() is not None
    compiled = lowered.compile()
    del lowered
    gc.collect()
    np.testing.assert_array_equal(
        compiled(jnp.ones(4, dtype=jnp.float32))[0], np.full(4, 4)
    )


def test_callback_captures_survive_asynchronous_executable_destruction(gpu):
    submitted = threading.Event()
    completed = gpu.cuda.Event(disable_timing=True)
    kernel = gpu.RawKernel(
        r"""
        extern "C" __global__ void delayed(const float* weights, float* out) {
          unsigned long long start = clock64();
          while (clock64() - start < 200000000) {}
          out[0] = weights[0];
        }
    """,
        "delayed",
    )
    kernel.compile()

    def make():
        weights = gpu.full(1, 7, dtype="float32")
        reference = weakref.ref(weights)

        def external(out, args, config):
            kernel((1,), (1,), (weights, out[0]))
            completed.record(jb.current_context().stream)
            submitted.set()

        op = jb.get_call(external, lambda x: ((x.shape, x.dtype),))
        return jax.jit(op).lower(
            jax.ShapeDtypeStruct((1,), np.float32)
        ).compile(), reference

    compiled, reference = make()
    x = jnp.zeros(1, dtype=jnp.float32)
    x.block_until_ready()
    result = compiled(x)[0]
    assert submitted.wait(5)
    del compiled
    jax.clear_caches()
    gc.collect()
    assert completed.done or reference() is not None
    np.testing.assert_array_equal(result, [7])
    jb.synchronize()
    gc.collect()
    assert reference() is None


def test_cached_resources_and_async_retention(gpu):
    created = []
    destroyed = []

    class Resource:
        def __init__(self):
            created.append(1)

        def __del__(self):
            destroyed.append(1)

    def copy(out, args, kwargs_dump):
        context = jb.current_context()
        resource = context.cached("resource", Resource)
        assert resource is not None
        assert context.device == args[0].device.id
        assert context.stream.ptr == gpu.cuda.get_current_stream().ptr
        gpu.copyto(out[0], args[0])

    def abstract(x):
        return ((x.shape, x.dtype),)

    op = jb.get_linear_call(copy, copy, abstract, abstract)
    x = jnp.arange(16.0)
    compiled = jax.jit(op)
    for _ in range(16):
        x = compiled(x)[0]
    x.block_until_ready()
    np.testing.assert_array_equal(x, np.arange(16))
    assert len(created) == 1
    assert not destroyed
    jb.synchronize(clear_cache=True)
    assert destroyed == [1]
    assert jb.runtime_stats()["pending"] == 0


def test_resources_survive_work_and_cache_eviction(gpu):
    completed = []
    kernel = gpu.RawKernel(
        r"""
    extern "C" __global__ void delayed_copy(const float* x, float* y) {
        unsigned long long begin = clock64();
        while (clock64() - begin < 5000000) {}
        y[0] = x[0];
    }""",
        "delayed_copy",
    )
    kernel.compile()

    class Resource:
        def __init__(self):
            self.event = gpu.cuda.Event(disable_timing=True)

        def __del__(self):
            completed.append(self.event.done)

    def execute(out, args, kwargs_dump):
        context = jb.current_context()
        key = jb.load_kwargs(kwargs_dump)["key"]
        resource = context.cached(key, Resource)
        temporary = gpu.array([key], dtype="float32")
        kernel((1,), (1,), (temporary, out[0]))
        resource.event.record(context.stream)
        context.retain(temporary)

    def abstract(x, **kwargs):
        return ((x.shape, x.dtype),)

    jb.configure_runtime(max_cache_entries=1, max_pending_calls=2)
    try:
        op = jb.get_call(execute, abstract)
        results = [
            jax.jit(lambda x, key=key: op(x, key=key)[0])(jnp.ones(1, dtype="float32"))
            for key in range(4)
        ]
        for key, result in enumerate(results):
            np.testing.assert_array_equal(result, [key])
        jb.synchronize(clear_cache=True)
        assert completed == [True] * 4
    finally:
        jb.configure_runtime()


def test_pending_scratch_obeys_byte_pressure_and_reuses_events(gpu):
    kernel = gpu.RawKernel(
        r"""
        extern "C" __global__ void wait_copy(const float* x, float* y) {
          unsigned long long start = clock64();
          while (clock64() - start < 1000000) {}
          y[0] = x[0];
        }
        """,
        "wait_copy",
    )
    kernel.compile()
    scratch_bytes = 1 << 20

    def external(out, args, config):
        scratch = gpu.empty(scratch_bytes // 4, dtype="float32")
        gpu.copyto(scratch[:1], args[0])
        kernel((1,), (1,), (scratch, out[0]))
        jb.current_context().retain(scratch)

    jb.configure_runtime(max_pending_bytes=scratch_bytes, max_pending_calls=256)
    try:
        op = jax.jit(jb.get_call(external, lambda x: ((x.shape, x.dtype),)))
        xs = [jnp.array([i], dtype=jnp.float32) for i in range(8)]
        outputs = [op(x)[0] for x in xs]
        jax.block_until_ready(outputs)
        assert jb.runtime_stats()["pending_bytes"] <= scratch_bytes
        for i, output in enumerate(outputs):
            np.testing.assert_array_equal(output, [i])
        jb.synchronize()
        before = jb.runtime_stats()["event_creations"]
        for x in xs:
            op(x)[0].block_until_ready()
            jb.synchronize()
        assert jb.runtime_stats()["event_creations"] == before
        assert jb.runtime_stats()["pending_bytes"] == 0
    finally:
        jb.configure_runtime()
