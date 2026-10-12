"""Return-only GPU functions can borrow output allocations without changing code."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


def test_returned_allocation_is_the_xla_output(gpu):
    pointers = []

    def external(x):
        value = gpu.square(x)
        pointers.append(value.data.ptr)
        return value

    compiled = jax.jit(jb.bind(external, out="allocate"))
    inputs = [jnp.arange(32, dtype=jnp.float32) + i for i in range(3)]
    outputs = [compiled(x) for x in inputs]
    jax.block_until_ready(outputs)
    assert pointers == [y.unsafe_buffer_pointer() for y in outputs]
    for i, value in enumerate(outputs):
        np.testing.assert_array_equal(value, (np.arange(32) + i) ** 2)


@pytest.mark.parametrize("reorder", [False, True])
def test_multiple_outputs_and_copy_cycles(gpu, reorder):
    def external(x):
        first, second = x + 1, x + 2
        return (second, first) if reorder else (first, second)

    def output(x):
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return spec, spec

    x = jnp.arange(12, dtype=jnp.float32)
    actual = jax.jit(jb.bind(external, output=output, out="allocate"))(x)
    for value, offset in zip(actual, (2, 1) if reorder else (1, 2)):
        np.testing.assert_array_equal(value, np.arange(12) + offset)


@pytest.mark.parametrize("view", ["transpose", "reverse", "slice"])
def test_returned_views_use_safe_fallback(gpu, view):
    def external(x):
        value = gpu.square(x)
        if view == "transpose":
            return value.T
        if view == "reverse":
            return value[::-1]
        return value[1:]

    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    spec = jax.ShapeDtypeStruct((3, 4) if view == "slice" else (4, 4), np.float32)
    actual = jax.jit(jb.bind(external, output=spec, out="allocate"))(x)
    expected = np.arange(16, dtype=np.float32).reshape(4, 4) ** 2
    expected = (
        expected.T
        if view == "transpose"
        else expected[::-1]
        if view == "reverse"
        else expected[1:]
    )
    np.testing.assert_array_equal(actual, expected)


def test_scratch_allocation_does_not_replace_result(gpu):
    def external(x):
        scratch = gpu.empty_like(x)
        scratch.fill(-1)
        return x * 3

    actual = jax.jit(jb.bind(external, out="allocate"))(
        jnp.arange(16, dtype=jnp.float32)
    )
    np.testing.assert_array_equal(actual, 3 * np.arange(16))


def test_forwarded_output_uses_no_owned_cupy_allocation(gpu):
    owned_bytes = []

    def external(x):
        pool = gpu.get_default_memory_pool()
        before = pool.used_bytes()
        value = gpu.square(x)
        owned_bytes.append(pool.used_bytes() - before)
        return value

    x = jnp.arange(1024, dtype=jnp.float32)
    actual = jax.jit(jb.bind(external, out="allocate"))(x)
    np.testing.assert_array_equal(actual, np.arange(1024) ** 2)
    assert owned_bytes == [0]
    np.testing.assert_array_equal(x, np.arange(1024))


@pytest.mark.parametrize("shape", [(), (0,), (0, 4), (3, 2, 4)])
def test_scalar_empty_and_higher_rank_outputs(gpu, shape):
    host = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
    actual = jax.jit(jb.bind(gpu.square, out="allocate"))(jnp.asarray(host))
    np.testing.assert_array_equal(actual, host**2)


def test_allocation_forwarding_keeps_result_validation(gpu):
    x = jnp.arange(4, dtype=jnp.float32)
    for external, message in (
        (lambda x: np.ones(4, dtype=np.float32), "GPU"),
        (lambda x: gpu.ones(5, dtype=np.float32), "shape"),
        (lambda x: gpu.ones(4, dtype=np.int32), "dtype"),
        (lambda x: {"wrong": x}, "structure"),
    ):
        with pytest.raises(Exception, match=message):
            jax.jit(jb.bind(external, out="allocate"))(x).block_until_ready()


def test_persistent_cache_allocates_outside_outputs(gpu):
    pointers = []

    def external(x):
        weights = jb.current_context().cached(
            "weights", lambda: gpu.full(x.shape, 7, dtype=x.dtype)
        )
        value = x + weights
        pointers.append((weights.data.ptr, value.data.ptr))
        return value

    compiled = jax.jit(jb.bind(external, out="allocate"))
    outputs = [compiled(jnp.full(16, i, dtype=jnp.float32)) for i in range(3)]
    jax.block_until_ready(outputs)
    assert len({pair[0] for pair in pointers}) == 1
    for i, value in enumerate(outputs):
        assert pointers[i][1] == value.unsafe_buffer_pointer()
        assert pointers[i][0] != pointers[i][1]
        np.testing.assert_array_equal(value, np.full(16, i + 7))
    jb.synchronize(clear_cache=True)


def test_cold_fft_plan_workspace_is_not_an_output_allocation(gpu):
    from cupyx.scipy.fft import fft, get_fft_plan

    pointers = []

    def external(x):
        plan = jb.current_context().cached("plan", lambda: get_fft_plan(x))
        value = fft(x, plan=plan)
        pointers.append(value.data.ptr)
        return value

    compiled = jax.jit(jb.bind(external, out="allocate"))
    # An impulse has a constant Fourier transform, an analytic oracle with
    # no cancellation tolerance. Every plan is created inside the first call.
    outputs = []
    for amplitude in (1, 3, 2):
        host = np.zeros(65536, np.complex64)
        host[0] = amplitude
        value = compiled(jnp.asarray(host))
        np.testing.assert_array_equal(value, np.full(host.shape, amplitude))
        outputs.append(value)
    assert pointers == [value.unsafe_buffer_pointer() for value in outputs]
    jb.synchronize(clear_cache=True)


def test_allocation_forwarding_with_workspace_graph_and_derivatives(gpu):
    op = jb.bind(
        lambda x: gpu.square(x),
        out="allocate",
        workspace_bytes=512,
        graph_compatible=True,
        jvp=lambda p, t: 2 * p[0] * t[0],
    )
    x = jnp.arange(16, dtype=jnp.float32)
    compiled = jax.jit(
        op,
        compiler_options={
            "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
            "xla_gpu_graph_min_graph_size": 1,
        },
    )
    for offset in range(3):
        np.testing.assert_array_equal(
            compiled(x + offset), (np.arange(16) + offset) ** 2
        )
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: op(x).sum()))(x), 2 * np.arange(16)
    )
    np.testing.assert_array_equal(
        jax.jit(jax.jacfwd(jax.grad(lambda x: op(x).sum())))(x), 2 * np.eye(16)
    )


def test_allocation_mode_rejects_incompatible_options():
    with pytest.raises(ValueError, match="CuPy"):
        jb.bind(lambda x: x, out="allocate", backend="torch")
    with pytest.raises(ValueError, match="aliases"):
        jb.bind(lambda x: x, out="allocate", input_output_aliases={0: 0})
    with pytest.raises(ValueError, match="out"):
        jb.bind(lambda x: x, out="typo")
