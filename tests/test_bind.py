"""The normal user interface is a single factory for ordinary GPU functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


class _Settings:
    def __init__(self, factor):
        self.factor = factor


def shape(x, **kwargs):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def test_eager_binding_reuses_shape_trace_and_static_configuration(gpu):
    traces = []

    def output(x, *, factor):
        traces.append((x.shape, factor))
        return shape(x)

    op = jb.bind(
        lambda x, *, factor: x * factor, output, jvp=lambda p, t, factor: factor * t[0]
    )
    x = jnp.arange(16, dtype=jnp.float32)
    for factor in (2, 2, 3, 2):
        np.testing.assert_array_equal(op(x, factor=factor), np.arange(16) * factor)
    assert traces == [((16,), 2), ((16,), 3)]
    np.testing.assert_array_equal(
        jax.grad(lambda x: op(x, factor=2).sum())(x),
        np.full(16, 2),
    )


def test_mutable_static_keywords_are_decoded_fresh_for_each_callback(gpu):
    def external(x, *, settings, out):
        factor = settings.factor
        settings.factor += 1
        gpu.multiply(x, factor, out=out)

    op = jb.bind(external, output=lambda x, **kw: shape(x), out=True)
    settings = _Settings(2)
    x = jnp.arange(16, dtype=jnp.float32)
    compiled = jax.jit(lambda x: op(x, settings=settings))
    for _ in range(4):
        np.testing.assert_array_equal(compiled(x), 2 * np.arange(16))
    assert settings.factor == 2


def test_bind_ordinary_cupy_function_with_jvp(gpu):
    op = jb.bind(gpu.square, output=shape, jvp=lambda p, t: 2 * p[0] * t[0])
    x = jnp.arange(6, dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(op)(x), np.arange(6) ** 2)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: op(x).sum()))(x), 2 * np.arange(6)
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(jax.grad(lambda x: op(x).sum())))(x), 2 * np.eye(6)
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(op))(jnp.stack((x, 2 * x))),
        np.stack((np.arange(6) ** 2, 4 * np.arange(6) ** 2)),
    )


def test_bind_output_tree_static_keywords_and_gpu_pointers(gpu):
    observed = []

    def external(x, *, factor):
        observed.append((x.data.ptr, gpu.cuda.get_current_stream().ptr))
        return {"scaled": x * factor, "pair": (x + 1, x - 1)}

    def output(x, *, factor):
        spec = shape(x)
        return {"scaled": spec, "pair": (spec, spec)}

    op = jb.bind(external, output=output)
    x = jnp.arange(7, dtype=jnp.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        result = jax.jit(lambda x: op(x, factor=2.0))(x)
        jax.block_until_ready(result)
    assert observed[0][0] == x.unsafe_buffer_pointer()
    assert observed[0][1] != 0
    np.testing.assert_array_equal(result["scaled"], 2 * np.arange(7))
    np.testing.assert_array_equal(result["pair"][0], np.arange(7) + 1)


def test_bind_shape_changing_linear_transpose(gpu):
    op = jb.bind(
        lambda x: x.reshape(2, 3),
        output=jax.ShapeDtypeStruct((2, 3), np.float32),
        transpose=lambda cot: cot.reshape(6),
    )
    x = jnp.arange(6, dtype=jnp.float32)
    actual = jax.jit(jax.grad(lambda x: jnp.sum(op(x) ** 2)))(x)
    np.testing.assert_array_equal(actual, 2 * np.arange(6))


def test_bind_multi_input_linear_transpose(gpu):
    def external(x, y):
        return 2 * x + 3 * y

    op = jb.bind(
        external, output=lambda x, y: shape(x), transpose=lambda cot: (2 * cot, 3 * cot)
    )
    x = jnp.arange(6, dtype=jnp.float32)
    actual = jax.jit(jax.grad(lambda x, y: op(x, y).sum(), argnums=(0, 1)))(x, x)
    np.testing.assert_array_equal(actual[0], np.full(6, 2))
    np.testing.assert_array_equal(actual[1], np.full(6, 3))


def test_bind_torch_function_and_jvp(gpu):
    torch = pytest.importorskip("torch")
    seen = []

    def external(x):
        seen.append((x.data_ptr(), torch.cuda.current_stream().cuda_stream))
        return torch.square(x)

    op = jb.bind(
        external, output=shape, backend="torch", jvp=lambda p, t: 2 * p[0] * t[0]
    )
    x = jnp.arange(8, dtype=jnp.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        actual = jax.jit(op)(x).block_until_ready()
    assert seen[0][0] == x.unsafe_buffer_pointer()
    assert seen[0][1] != 0
    np.testing.assert_array_equal(actual, np.arange(8) ** 2)
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: op(x).sum()))(x), 2 * np.arange(8)
    )


def test_dlpack_waits_when_callback_changes_producer_stream(gpu):
    torch = pytest.importorskip("torch")
    kernel = gpu.RawKernel(
        'extern "C" __global__ void delayed(const float* x, float* y) {'
        " unsigned long long start = clock64();"
        " while (clock64() - start < 1000000) {}"
        " y[threadIdx.x] = x[threadIdx.x] + 1; }",
        "delayed",
    )
    kernel.compile()

    def external(x, *, out):
        context = jb.current_context()
        other = gpu.cuda.Stream(non_blocking=True)
        consumer = torch.cuda.ExternalStream(context.stream.ptr, device=context.device)
        with other, torch.cuda.stream(consumer):
            kernel((1,), (16,), (x, out))
            tensor = torch.utils.dlpack.from_dlpack(out)
            tensor.mul_(3)
        context.retain(other, consumer, tensor)

    op = jax.jit(jb.bind(external, out=True))
    host = np.arange(16, dtype=np.float32)
    x = jnp.asarray(host)
    for _ in range(4):
        np.testing.assert_array_equal(op(x), 3 * (host + 1))


def test_bind_rejects_invalid_rules():
    with pytest.raises(ValueError, match="jvp.*transpose"):
        jb.bind(lambda x: x, output=shape, jvp=lambda p, t: t[0], transpose=lambda x: x)
    with pytest.raises(ValueError, match="backend"):
        jb.bind(lambda x: x, output=shape, backend="cpu")
    with pytest.raises(ValueError, match="workspace_bytes"):
        jb.bind(lambda x: x, output=shape, workspace_bytes=-1)
    with pytest.raises(ValueError, match="CuPy"):
        jb.bind(lambda x: x, output=shape, backend="torch", workspace_bytes=1024)
    with pytest.raises(TypeError, match="ShapeDtypeStruct"):
        jax.eval_shape(
            jb.bind(lambda x: x, output=(2, np.float32)),
            jax.ShapeDtypeStruct((2,), np.float32),
        )


def test_bind_rejects_host_results_and_wrong_specs(gpu):
    x = jnp.arange(4, dtype=jnp.float32)
    for external, message in (
        (lambda x: np.ones(4, dtype="float32"), "GPU"),
        (lambda x: gpu.ones(5, dtype="float32"), "shape"),
        (lambda x: {"wrong": x}, "structure"),
    ):
        op = jb.bind(external, output=shape)
        with pytest.raises(Exception, match=message):
            jax.jit(op)(x).block_until_ready()


def test_direct_output_tree_and_trace_only_shape_metadata(gpu):
    def output(x):
        assert not isinstance(x, gpu.ndarray)
        return {"sum": shape(x), "square": shape(x)}

    def external(x, *, out):
        gpu.add(x, 2, out=out["sum"])
        gpu.square(x, out=out["square"])

    op = jb.bind(external, output=output, out=True)
    x = jnp.arange(8, dtype=jnp.float32)
    result = jax.jit(op)(x)
    np.testing.assert_array_equal(result["sum"], np.arange(8) + 2)
    np.testing.assert_array_equal(result["square"], np.arange(8) ** 2)


def test_output_alias_with_different_strides_is_rejected(gpu):
    def external(x, *, out):
        gpu.copyto(out, x)
        return out.T

    op = jb.bind(external, output=shape, out=True)
    with pytest.raises(Exception, match="strides"):
        jax.jit(op)(jnp.arange(9, dtype=jnp.float32).reshape(3, 3)).block_until_ready()


def test_torch_direct_output(gpu):
    torch = pytest.importorskip("torch")
    op = jb.bind(
        lambda x, *, out: torch.square(x, out=out),
        output=shape,
        out=True,
        backend="torch",
    )
    np.testing.assert_array_equal(
        jax.jit(op)(jnp.arange(8, dtype=jnp.float32)), np.arange(8) ** 2
    )


def test_direct_output_can_use_donated_input_storage(gpu):
    seen = []

    def external(x, *, out):
        seen.append((x.data.ptr, out.data.ptr))
        gpu.square(x, out=out)

    op = jb.bind(
        external,
        output=shape,
        out=True,
        input_output_aliases={0: 0},
        jvp=lambda p, t: 2 * p[0] * t[0],
    )
    x = jnp.arange(1024, dtype=jnp.float32)
    pointer = x.unsafe_buffer_pointer()
    compiled = jax.jit(op, donate_argnums=(0,)).lower(x).compile()
    assert compiled.memory_analysis().alias_size_in_bytes == x.nbytes
    y = compiled(x).block_until_ready()
    assert seen[-1] == (pointer, pointer)
    assert y.unsafe_buffer_pointer() == pointer
    np.testing.assert_array_equal(y, np.arange(1024) ** 2)
    gradient = jax.jit(jax.grad(lambda x: op(x).sum()))(
        jnp.arange(8, dtype=jnp.float32)
    )
    np.testing.assert_array_equal(gradient, 2 * np.arange(8))


def test_local_custom_output_tree_needs_no_pickling(gpu):
    @jax.tree_util.register_pytree_node_class
    class LocalResult:
        def __init__(self, value):
            self.value = value

        def tree_flatten(self):
            return (self.value,), None

        @classmethod
        def tree_unflatten(cls, metadata, values):
            return cls(values[0])

    def external(x, *, out):
        gpu.negative(x, out=out.value)

    op = jb.bind(external, output=lambda x: LocalResult(shape(x)), out=True)
    actual = jax.jit(op)(jnp.arange(8, dtype=jnp.float32))
    assert isinstance(actual, LocalResult)
    np.testing.assert_array_equal(actual.value, -np.arange(8))


def test_bind_defaults_trace_without_external_execution():
    def forbidden(*args, **kwargs):
        raise AssertionError("Shape inference must not execute GPU code")

    op = jb.bind(forbidden, jvp=lambda p, t: 2 * p[0] * t[0])
    x = jax.ShapeDtypeStruct((3, 7), np.float32)
    value = jax.eval_shape(op, x)
    gradient = jax.eval_shape(jax.grad(lambda x: op(x).sum()), x)
    assert value.shape == gradient.shape == x.shape
    assert value.dtype == gradient.dtype == x.dtype


def test_bind_no_input_requires_explicit_output():
    with pytest.raises(ValueError, match="output"):
        jax.eval_shape(jb.bind(lambda: None))


def test_bind_default_output_and_direct_gpu_storage(gpu):
    seen = []

    def square(x, *, out):
        seen.append((x.data.ptr, out.data.ptr))
        gpu.square(x, out=out)

    op = jb.bind(square, out=True, jvp=lambda p, t: 2 * p[0] * t[0])
    x = jnp.arange(6, dtype=jnp.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(op)(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: op(x).sum()))(x).block_until_ready()
    assert seen[0] == (x.unsafe_buffer_pointer(), value.unsafe_buffer_pointer())
    np.testing.assert_array_equal(value, np.arange(6) ** 2)
    np.testing.assert_array_equal(gradient, 2 * np.arange(6))


@pytest.mark.parametrize("derivative", ["jvp", "transpose"])
def test_bind_retracing_shapes_and_static_keywords(gpu, derivative):
    def external(x, *, columns, out):
        gpu.copyto(out, x[:, :columns])

    def transpose(cotangent, *, columns, out):
        out.fill(0)
        gpu.copyto(out[:, :columns], cotangent)

    def output(x, *, columns):
        return jax.ShapeDtypeStruct((x.shape[0], columns), x.dtype)

    options = (
        {"jvp": lambda p, t, columns: t[0][:, :columns]}
        if derivative == "jvp"
        else {"transpose": transpose}
    )
    op = jb.bind(external, output, out=True, **options)
    compiled = jax.jit(op, static_argnames="columns")
    grad = jax.jit(
        jax.grad(lambda x, columns: op(x, columns=columns).sum()),
        static_argnames="columns",
    )
    for shape, columns in (((2, 5), 3), ((4, 7), 2), ((2, 5), 4), ((2, 5), 3)):
        host = np.arange(np.prod(shape), dtype="float32").reshape(shape)
        x = jnp.asarray(host)
        with jax.transfer_guard_device_to_host("disallow"):
            value = compiled(x, columns=columns).block_until_ready()
            gradient = grad(x, columns=columns).block_until_ready()
        np.testing.assert_array_equal(value, host[:, :columns])
        expected_gradient = np.zeros(shape, dtype="float32")
        expected_gradient[:, :columns] = 1
        np.testing.assert_array_equal(gradient, expected_gradient)


def test_bind_default_output_requires_matching_result(gpu):
    op = jb.bind(gpu.sum)
    with pytest.raises(Exception, match="shape/dtype.*output"):
        jax.jit(op)(jnp.arange(4, dtype=jnp.float32)).block_until_ready()
