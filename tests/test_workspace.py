"""XLA owns scratch lifetimes; external functions retain their algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


def test_workspace_routes_temporary_allocations_and_preserves_ad(gpu):
    pointers = []

    def external(x, *, out):
        scratch = gpu.square(x)
        context = jb.current_context()
        pointers.append(
            (scratch.data.ptr, context.workspace.data.ptr, context.workspace.nbytes)
        )
        gpu.add(scratch, 1, out=out)
        context.retain(scratch)

    op = jb.bind(
        external,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        out=True,
        workspace_bytes=lambda x: x.nbytes + 512,
        jvp=lambda p, t: 2 * p[0] * t[0],
    )
    x = jnp.arange(1024, dtype=jnp.float32)
    value = jax.jit(op)(x)
    np.testing.assert_array_equal(value, np.arange(1024) ** 2 + 1)
    pointer, base, size = pointers[-1]
    assert base <= pointer < base + size
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: op(x).sum()))(x), 2 * np.arange(1024)
    )
    np.testing.assert_array_equal(
        jax.jit(jax.jacfwd(jax.grad(lambda x: op(x).sum())))(x[:4]), 2 * np.eye(4)
    )
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(op))(jnp.stack((x, 2 * x))),
        np.stack((np.arange(1024) ** 2 + 1, 4 * np.arange(1024) ** 2 + 1)),
    )


def test_cached_resources_are_allocated_outside_reusable_workspace(gpu):
    allocations = []

    def external(x, *, out):
        context = jb.current_context()
        weights = context.cached("weights", lambda: gpu.full_like(x, 2))
        allocations.append(
            (weights.data.ptr, context.workspace.data.ptr, context.workspace.nbytes)
        )
        value = gpu.square(x)
        gpu.multiply(value, weights, out=out)
        context.retain(value)

    op = jb.bind(
        external,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        out=True,
        workspace_bytes=4096,
        graph_compatible=True,
    )
    options = {
        "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
        "xla_gpu_graph_min_graph_size": 1,
    }
    compiled = jax.jit(op, compiler_options=options)
    for i in range(4):
        x = jnp.arange(1024, dtype=jnp.float32) + i
        np.testing.assert_array_equal(compiled(x), 2 * (np.arange(1024) + i) ** 2)
    assert all(not base <= pointer < base + size for pointer, base, size in allocations)


def test_workspace_reuses_freed_temporaries_and_reports_exhaustion(gpu):
    def external(x, *, out):
        for _ in range(16):
            a = gpu.square(x)
            gpu.copyto(out, a)
            del a

    def shape(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    x = jnp.arange(1024, dtype=jnp.float32)
    op = jb.bind(external, output=shape, out=True, workspace_bytes=4096)
    np.testing.assert_array_equal(jax.jit(op)(x), np.arange(1024) ** 2)
    too_small = jb.bind(external, output=shape, out=True, workspace_bytes=512)
    with pytest.raises(Exception, match="workspace.*exhausted"):
        jax.jit(too_small)(x).block_until_ready()


def test_workspace_graph_loop_has_bounded_external_memory(gpu):
    def external(x, *, out):
        a = gpu.empty_like(x)
        gpu.add(x, 1, out=a)
        gpu.copyto(out, a)
        jb.current_context().retain(a)

    op = jb.bind(
        external,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        out=True,
        workspace_bytes=4096,
        graph_compatible=True,
    )
    x = jnp.zeros(1024, dtype=jnp.float32)
    jax.jit(op)(x).block_until_ready()
    pool = gpu.get_default_memory_pool()
    before = pool.used_bytes()
    compiled = (
        jax.jit(
            lambda x: jax.lax.fori_loop(0, 32, lambda i, x: op(x), x, unroll=True),
            compiler_options={
                "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
                "xla_gpu_graph_min_graph_size": 1,
            },
        )
        .lower(x)
        .compile()
    )
    for i in range(8):
        np.testing.assert_array_equal(compiled(x + i), np.full(1024, 32 + i))
    jb.synchronize()
    assert pool.used_bytes() <= before + 4096
    assert compiled.memory_analysis().temp_size_in_bytes >= 4096


@pytest.mark.parametrize("name", ["sort", "cumsum"])
def test_return_value_operations_use_workspace_under_graph_replay(gpu, name):
    function = getattr(gpu, name)
    op = jb.bind(
        function,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        workspace_bytes=1 << 20,
        graph_compatible=True,
    )
    options = {
        "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
        "xla_gpu_graph_min_graph_size": 1,
    }
    compiled = jax.jit(op, compiler_options=options)
    rng = np.random.default_rng(190)
    for _ in range(4):
        x = rng.uniform(size=4096).astype("float32")
        expected = np.sort(x) if name == "sort" else np.cumsum(x.astype("float64"))
        np.testing.assert_allclose(
            compiled(jnp.asarray(x)), expected, rtol=3e-6, atol=1e-6
        )
