"""Graph replay must preserve GPU values and the allocations it references."""

import gc
import weakref
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


OPTIONS = {
    "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
    "xla_gpu_graph_min_graph_size": 1,
}


def test_python_graph_replays_with_live_scratch_and_releases_executable(gpu):
    calls = []
    scratch_refs = []

    def external(out, args, config):
        calls.append(1)
        scratch = gpu.empty_like(args[0])
        scratch_refs.append(weakref.ref(scratch))
        gpu.multiply(args[0], 2, out=scratch)
        gpu.add(scratch, 1, out=out[0])
        jb.current_context().retain(scratch)

    op = jb.get_call(external, lambda x: ((x.shape, x.dtype),), graph_compatible=True)
    # Load/compile the external kernels and allocate scratch outside capture.
    warm = jax.jit(op)(jnp.ones(4096, dtype=jnp.float32))[0]
    warm.block_until_ready()
    jb.synchronize()
    calls.clear()
    scratch_refs.clear()
    compiled = (
        jax.jit(op, compiler_options=OPTIONS)
        .lower(jax.ShapeDtypeStruct((4096,), np.float32))
        .compile()
    )
    for i in range(12):
        x = jnp.full((4096,), i, dtype=jnp.float32)
        with jax.transfer_guard_device_to_host("disallow"):
            y = compiled(x)[0].block_until_ready()
        np.testing.assert_array_equal(y, np.full((4096,), 2 * i + 1))
        del x, y
    assert len(calls) < 12, "The operation executed in Python on every replay"
    jb.synchronize()
    assert any(ref() is not None for ref in scratch_refs)
    del compiled
    jax.clear_caches()
    for _ in range(200):
        jb.synchronize()
        gc.collect()
        if all(ref() is None for ref in scratch_refs):
            break
        time.sleep(0.01)
    assert all(ref() is None for ref in scratch_refs)


def test_bind_writes_directly_to_output(gpu):
    seen = []

    def external(x, *, out, factor):
        seen.append(out.data.ptr)
        gpu.multiply(x, factor, out=out)
        return out

    op = jb.bind(
        external,
        output=lambda x, **kw: jax.ShapeDtypeStruct(x.shape, x.dtype),
        out=True,
    )
    x = jnp.arange(1024, dtype=jnp.float32)
    y = jax.jit(lambda x: op(x, factor=3))(x).block_until_ready()
    assert seen[-1] == y.unsafe_buffer_pointer()
    np.testing.assert_array_equal(y, np.arange(1024) * 3)


def test_obsolete_graph_captures_release_scratch(gpu):
    refs = []
    pointers = []

    def external(out, args, config):
        scratch = gpu.empty_like(args[0])
        refs.append(weakref.ref(scratch))
        pointers.append(scratch.data.ptr)
        gpu.multiply(args[0], 2, out=scratch)
        gpu.copyto(out[0], scratch)
        jb.current_context().retain(scratch)

    op = jb.get_call(external, lambda x: ((x.shape, x.dtype),), graph_compatible=True)
    compiled = jax.jit(op, compiler_options=OPTIONS)
    inputs = [jnp.full((1024,), i, dtype=jnp.float32) for i in range(32)]
    outputs = [compiled(x)[0] for x in inputs]
    jax.block_until_ready(outputs)
    for i, output in enumerate(outputs):
        np.testing.assert_array_equal(output, np.full((1024,), 2 * i))
    # Changing buffers causes recapture. Obsolete graphs must not accumulate
    # scratch for the entire lifetime of this still-live executable.
    for _ in range(200):
        jb.synchronize()
        if sum(ref() is not None for ref in refs) <= 65:
            break
        time.sleep(0.01)
    assert len(refs) >= 16, "This workload did not exercise graph updates"
    # The binding keeps at most 64 recordings plus the uncaptured warmup.
    # Their array views share one allocation instead of owning scratch each.
    assert sum(ref() is not None for ref in refs) <= 65
    assert len(set(pointers[1:])) == 1

    more = [jnp.full((1024,), 32 + i, dtype=jnp.float32) for i in range(32)]
    more_outputs = [compiled(x)[0] for x in more]
    jax.block_until_ready(more_outputs)
    for i, output in enumerate(more_outputs):
        np.testing.assert_array_equal(output, np.full((1024,), 2 * (32 + i)))
    for _ in range(200):
        jb.synchronize()
        if sum(ref() is not None for ref in refs) <= 65:
            break
        time.sleep(0.01)
    assert sum(ref() is not None for ref in refs) <= 65
    assert len(set(pointers[1:])) == 1


def test_graph_owns_implicit_temporaries_and_preserves_transformations(gpu):
    def external(x, *, out):
        scratch = gpu.empty_like(x)
        gpu.square(x, out=scratch)
        gpu.add(scratch, 1, out=out)

    op = jb.bind(
        external,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        out=True,
        graph_compatible=True,
        jvp=lambda p, t: 2 * p[0] * t[0],
    )
    x = jnp.arange(24, dtype=jnp.float32).reshape(3, 8)
    compiled = jax.jit(
        jax.vmap(op),
        compiler_options={
            **OPTIONS,
            "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
        },
    )
    for scale in (1, 2, 3, 4):
        actual = compiled(scale * x)
        np.testing.assert_array_equal(
            actual, (scale * np.arange(24).reshape(3, 8)) ** 2 + 1
        )
    gradient = jax.jit(jax.grad(lambda x: op(x).sum()), compiler_options=OPTIONS)
    np.testing.assert_array_equal(gradient(x), 2 * np.arange(24).reshape(3, 8))
    hessian = jax.jit(
        jax.jacfwd(jax.grad(lambda x: op(x).sum())), compiler_options=OPTIONS
    )
    np.testing.assert_array_equal(hessian(x[0]), 2 * np.eye(8))


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_graph_reuses_recording_after_xla_trace_cache_eviction(gpu, options):
    calls = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void increment(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i] + 1; }",
        "increment",
    )

    def external(x, *, out):
        calls.append(1)
        kernel((1,), (16,), (x, out))

    # Compile the library kernel independently before capture. Donation gives
    # a deterministic set of 32 buffer addresses, larger than XLA's trace cache.
    warm = gpu.zeros(16, dtype=gpu.float32)
    kernel((1,), (16,), (warm, warm))
    gpu.cuda.Device().synchronize()
    op = jb.bind(
        external,
        out=True,
        graph_compatible=True,
        input_output_aliases={0: 0},
    )
    compiled = jax.jit(op, donate_argnums=(0,), compiler_options=options)
    values = [jnp.full(16, i, dtype=jnp.float32) for i in range(32)]
    values = [compiled(x) for x in values]
    jax.block_until_ready(values)
    # XLA can execute the first call before enabling capture. Populate its
    # recording too, then test a complete pass through all recurring addresses.
    values = [compiled(x) for x in values]
    jax.block_until_ready(values)
    recorded = len(calls)
    with jax.transfer_guard_device_to_host("disallow"):
        values = [compiled(x) for x in values]
        jax.block_until_ready(values)
    for i, value in enumerate(values):
        np.testing.assert_array_equal(value, np.full(16, i + 3))
    assert len(calls) == recorded, "Trace-cache eviction reran the Python operation"


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_recording_cache_is_bounded_and_retires_with_executable(gpu, options):
    contexts = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void increment(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i] + 1; }",
        "increment",
    )
    kernel.compile()

    def external(x, *, out):
        contexts.append(weakref.ref(jb.current_context()))
        kernel((1,), (16,), (x, out))

    op = jb.bind(external, out=True, graph_compatible=True, input_output_aliases={0: 0})
    compiled = jax.jit(op, donate_argnums=(0,), compiler_options=options)
    values = [jnp.full(16, i, dtype=jnp.float32) for i in range(96)]
    values = [compiled(x) for x in values]
    jax.block_until_ready(values)
    recorded = len(contexts)
    # Recently used recordings remain cached even after 96 distinct buffers.
    values[-32:] = [compiled(x) for x in values[-32:]]
    jax.block_until_ready(values)
    assert len(contexts) == recorded
    for i, value in enumerate(values):
        np.testing.assert_array_equal(value, np.full(16, i + 1 + (i >= 64)))
    for _ in range(200):
        jb.synchronize()
        if sum(ref() is not None for ref in contexts) <= 65:
            break
        time.sleep(0.01)
    assert sum(ref() is not None for ref in contexts) <= 65
    del compiled
    jax.clear_caches()
    for _ in range(200):
        jb.synchronize()
        gc.collect()
        if all(ref() is None for ref in contexts):
            break
        time.sleep(0.01)
    assert all(ref() is None for ref in contexts)


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_cached_recording_preserves_surrounding_gpu_work(gpu, options):
    kernel = gpu.RawKernel(
        'extern "C" __global__ void increment(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i] + 1; }",
        "increment",
    )
    kernel.compile()

    def external(x, *, out):
        kernel((1,), (16,), (x, out))

    op = jb.bind(external, out=True, graph_compatible=True)
    compiled = jax.jit(lambda x: 5 * op(3 * x) + 7, compiler_options=options)
    inputs = [jnp.full(16, i, dtype=jnp.float32) for i in range(32)]
    for _ in range(3):
        outputs = [compiled(x) for x in inputs]
        jax.block_until_ready(outputs)
        for i, output in enumerate(outputs):
            np.testing.assert_array_equal(output, np.full(16, 15 * i + 12))


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_graph_does_not_cache_opaque_memory_owners(gpu, options):
    refs = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void copy(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i]; }",
        "copy",
    )
    kernel.compile()

    def external(x, *, out):
        # UnownedMemory can wrap an external GPU allocation. Its owner is not
        # necessarily borrowed XLA storage, even though CuPy does not own it.
        owner = gpu.cuda.Memory(x.nbytes)
        memory = gpu.cuda.UnownedMemory(owner.ptr, x.nbytes, owner, device_id=0)
        scratch = gpu.ndarray(
            x.shape, x.dtype, memptr=gpu.cuda.MemoryPointer(memory, 0)
        )
        refs.append(weakref.ref(scratch))
        jb.current_context().retain(scratch)
        kernel((1,), (16,), (x, scratch))
        kernel((1,), (16,), (scratch, out))

    compiled = jax.jit(
        jb.bind(external, out=True, graph_compatible=True), compiler_options=options
    )
    inputs = [jnp.full(16, i, dtype=jnp.float32) for i in range(96)]
    outputs = [compiled(x) for x in inputs]
    jax.block_until_ready(outputs)
    for i, output in enumerate(outputs):
        np.testing.assert_array_equal(output, np.full(16, i))
    for _ in range(200):
        jb.synchronize()
        if sum(ref() is not None for ref in refs) <= 20:
            break
        time.sleep(0.01)
    assert sum(ref() is not None for ref in refs) <= 20


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_recordings_share_temporary_storage_and_skip_python(gpu, options):
    calls = []
    scratch_pointers = []

    def external(x, *, out):
        calls.append(1)
        scratch = gpu.empty_like(x)
        scratch_pointers.append(scratch.data.ptr)
        gpu.multiply(x, 2, out=scratch)
        gpu.add(scratch, 1, out=out)
        jb.current_context().retain(scratch)

    # Warm the library kernels outside capture. Donation fixes the address set
    # while each replay changes values, so a recording cannot cache answers.
    warm = gpu.ones(1024, dtype=gpu.float32)
    gpu.multiply(warm, 2, out=warm)
    gpu.add(warm, 1, out=warm)
    gpu.cuda.Device().synchronize()
    compiled = jax.jit(
        jb.bind(external, out=True, graph_compatible=True, input_output_aliases={0: 0}),
        donate_argnums=(0,),
        compiler_options=options,
    )
    values = [jnp.full(1024, i, dtype=jnp.float32) for i in range(32)]
    for _ in range(2):
        values = [compiled(x) for x in values]
        jax.block_until_ready(values)
    recorded = len(calls)
    recorded_pointers = set(scratch_pointers[2:])
    values = [compiled(x) for x in values]
    jax.block_until_ready(values)
    assert len(calls) == recorded, "Temporary storage prevented callback reuse"
    assert len(recorded_pointers) == 1, "Recordings duplicated temporary storage"
    for i, value in enumerate(values):
        np.testing.assert_array_equal(value, np.full(1024, 8 * i + 7))


def test_torch_direct_output_recordings_skip_python(gpu):
    torch = pytest.importorskip("torch")
    calls = []

    def external(x, *, out):
        calls.append(1)
        torch.add(x, 1, out=out)

    compiled = jax.jit(
        jb.bind(
            external,
            backend="torch",
            out=True,
            graph_compatible=True,
            input_output_aliases={0: 0},
        ),
        donate_argnums=(0,),
    )
    values = [jnp.full(16, i, dtype=jnp.float32) for i in range(32)]
    for _ in range(2):
        values = [compiled(x) for x in values]
        jax.block_until_ready(values)
    recorded = len(calls)
    values = [compiled(x) for x in values]
    jax.block_until_ready(values)
    assert len(calls) == recorded, "Borrowed Torch tensors prevented recording reuse"
    for i, value in enumerate(values):
        np.testing.assert_array_equal(value, np.full(16, i + 3))


def test_recordings_are_scoped_to_each_gpu(gpu):
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) < 2:
        pytest.skip("Run with two idle CUDA devices")
    calls = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void square(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i] * x[i]; }",
        "square",
    )

    def external(x, *, out):
        calls.append(jb.current_context().device)
        kernel((1,), (16,), (x, out))

    for ordinal in range(2):
        with gpu.cuda.Device(ordinal):
            warm = gpu.zeros(16, dtype=gpu.float32)
            kernel((1,), (16,), (warm, warm))
            gpu.cuda.Device().synchronize()
    compiled = jax.jit(
        jb.bind(external, out=True, graph_compatible=True, input_output_aliases={0: 0}),
        donate_argnums=(0,),
        compiler_options=OPTIONS,
    )
    values = [
        [
            jax.device_put(np.full(16, (i + 1) / 32, dtype=np.float32), device)
            for i in range(32)
        ]
        for device in devices[:2]
    ]
    # Devices can perform a different number of ordinary warmup calls before
    # XLA enables capture on its separate stream. Populate that stream's first
    # donated address too, then verify an entire pass without Python execution.
    for _ in range(3):
        values = [[compiled(x) for x in arrays] for arrays in values]
        jax.block_until_ready(values)
    recorded = len(calls)
    values = [[compiled(x) for x in arrays] for arrays in values]
    jax.block_until_ready(values)
    assert set(calls) == {0, 1}
    assert len(calls) == recorded
    for device, arrays in zip(devices, values):
        for i, value in enumerate(arrays):
            expected = np.full(16, (i + 1) / 32, dtype=np.float32)
            for _ in range(4):
                expected = np.square(expected)
            assert value.device == device
            np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("options", [OPTIONS, None])
def test_cached_library_plan_keeps_its_recording_stream(gpu, options):
    created = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void increment(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i] + 1; }",
        "increment",
    )
    kernel.compile()

    class Plan:
        def __init__(self):
            self.stream = jb.current_context().stream
            created.append(self.stream.ptr)

        def execute(self, x, out):
            kernel((1,), (16,), (x, out), stream=self.stream)

    def external(x, *, out):
        jb.current_context().cached("plan", Plan).execute(x, out)

    op = jb.bind(external, out=True, graph_compatible=True)
    for _ in range(2):
        compiled = jax.jit(lambda x: 3 * op(x + 5) + 7, compiler_options=options)
        inputs = [jnp.full(16, i, dtype=jnp.float32) for i in range(32)]
        for _ in range(3):
            outputs = [compiled(x) for x in inputs]
            jax.block_until_ready(outputs)
            for i, output in enumerate(outputs):
                np.testing.assert_array_equal(output, np.full(16, 3 * i + 25))
        del compiled
        jax.clear_caches()
        jb.synchronize()
    # XLA can use separate execution and capture streams. Each stream gets
    # one plan, including across executable destruction and recompilation.
    assert len(created) == len(set(created))


def test_managed_stream_retires_with_callback(gpu):
    from jaxbind_gpu import _bridge

    jb.synchronize()
    gc.collect()
    baseline = _bridge.recording_stream_count()
    kernel = gpu.RawKernel(
        'extern "C" __global__ void copy(const float* x, float* y) '
        "{ int i = threadIdx.x; y[i] = x[i]; }",
        "copy",
    )
    kernel.compile()

    def external(x, *, out):
        kernel((1,), (16,), (x, out))

    op = jb.bind(external, out=True, graph_compatible=True)
    compiled = jax.jit(op)
    np.testing.assert_array_equal(
        compiled(jnp.ones(16, dtype=jnp.float32)), np.ones(16)
    )
    assert _bridge.recording_stream_count() > baseline
    del compiled, op, external
    jax.clear_caches()
    for _ in range(200):
        jb.synchronize()
        gc.collect()
        if _bridge.recording_stream_count() <= baseline:
            break
        time.sleep(0.01)
    assert _bridge.recording_stream_count() <= baseline


def test_cached_launch_survives_immediate_binding_deletion(gpu):
    calls = []
    plans = []
    kernel = gpu.RawKernel(
        'extern "C" __global__ void increment(const float* x, float* y) '
        "{ unsigned long long start = clock64(); "
        "while (clock64() - start < 1000000ULL) {} "
        "int i = threadIdx.x; y[i] = x[i] + 1; }",
        "increment",
    )
    kernel.compile()

    class Plan:
        def __init__(self):
            self.kernel = kernel
            plans.append(weakref.ref(self))

        def execute(self, x, out):
            self.kernel((1,), (16,), (x, out))

    def external(x, *, out):
        calls.append(1)
        jb.current_context().cached("plan", Plan).execute(x, out)

    op = jb.bind(external, out=True, graph_compatible=True, input_output_aliases={0: 0})
    compiled = jax.jit(op, donate_argnums=(0,))
    value = jnp.zeros(16, dtype=jnp.float32)
    for _ in range(2):
        value = compiled(value).block_until_ready()
    recorded = len(calls)
    value = compiled(value)
    assert len(calls) == recorded
    del compiled, op, external
    jax.clear_caches()
    gc.collect()
    np.testing.assert_array_equal(value, np.full(16, 3))
    for _ in range(200):
        jb.synchronize()
        gc.collect()
        if all(ref() is None for ref in plans):
            break
        time.sleep(0.01)
    assert all(ref() is None for ref in plans)
