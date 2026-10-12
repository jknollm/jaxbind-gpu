"""Derivative expectations are analytic, independent of the GPU implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


@pytest.mark.parametrize("rules", ["jvp", "vjp", "both"])
def test_one_binding_supports_all_derivative_modes(gpu, rules):
    options = {}
    if rules in ("jvp", "both"):
        options["jvp"] = lambda p, t: 3 * p[0] ** 2 * t[0]
    if rules in ("vjp", "both"):
        options["vjp"] = lambda p, c: (3 * p[0] ** 2 * c,)
    cube = jb.bind(lambda x: x**3, **options)
    host = np.array([-1.5, 0.2, 2.0])
    x = jnp.asarray(host)
    direction = jnp.array([0.2, -0.3, 0.7])

    def calculation(x, direction):
        value, tangent = jax.jvp(cube, (x,), (direction,))
        _, pullback = jax.vjp(cube, x)
        return (
            value,
            tangent,
            pullback(direction)[0],
            jax.jacfwd(cube)(x),
            jax.jacrev(cube)(x),
            jax.hessian(lambda x: cube(x).sum())(x),
            jax.grad(lambda x: jax.grad(lambda y: cube(y).sum())(x).sum())(x),
        )

    with jax.transfer_guard_device_to_host("disallow"):
        result = jax.jit(calculation)(x, direction)
        jax.block_until_ready(result)
    expected = (
        host**3,
        3 * host**2 * np.asarray(direction),
        3 * host**2 * np.asarray(direction),
        np.diag(3 * host**2),
        np.diag(3 * host**2),
        np.diag(6 * host),
        6 * host,
    )
    for actual, wanted in zip(result, expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)


def test_vjp_shape_tracing_never_executes_external_code():
    def forbidden(*args, **kwargs):
        raise AssertionError("Tracing must not execute external code")

    op = jb.bind(forbidden, vjp=lambda p, c: (2 * p[0] * c,))
    x = jax.ShapeDtypeStruct((3, 7), np.float32)
    for rule in (op, jax.grad(lambda x: op(x).sum()), jax.jacfwd(op), jax.jacrev(op)):
        result = jax.eval_shape(rule, x)
        assert result.dtype == x.dtype


@pytest.mark.parametrize("rules", ["vjp", "both"])
def test_traceable_derivatives_have_analytic_values_without_external_execution(rules):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Pure derivative code must not execute the forward callback"
        )

    options = {"vjp": lambda p, c: (3 * p[0] ** 2 * c,)}
    if rules == "both":
        options["jvp"] = lambda p, t: 3 * p[0] ** 2 * t[0]
    cube = jb.bind(forbidden, **options)
    x = jnp.array([-1.5, 0.2, 2.0])
    expected = np.diag(3 * np.asarray(x) ** 2)
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(transform(cube))(x), expected, rtol=1e-12)
    hessian = jax.jit(jax.hessian(lambda x: cube(x).sum()))(x)
    np.testing.assert_allclose(hessian, np.diag(6 * np.asarray(x)), rtol=1e-12)


def test_explicit_gpu_jvp_and_vjp_need_no_derivative_of_the_callbacks(gpu):
    calls = []

    def push(x, direction):
        calls.append("jvp")
        return 2 * x * direction

    def pull(x, cotangent):
        calls.append("vjp")
        return 2 * x * cotangent

    # These derivative operations themselves are opaque, like an external library.
    push = jb.bind(push)
    pull = jb.bind(pull)
    op = jb.bind(
        gpu.square,
        out=True,
        jvp=lambda p, t: push(p[0], t[0]),
        vjp=lambda p, c: (pull(p[0], c),),
    )
    x = jnp.arange(5.0)
    with jax.transfer_guard_device_to_host("disallow"):
        gradient = jax.jit(jax.grad(lambda x: op(x).sum()))(x).block_until_ready()
    np.testing.assert_array_equal(gradient, 2 * np.arange(5))
    assert calls == ["vjp"]
    calls.clear()
    with jax.transfer_guard_device_to_host("disallow"):
        _, tangent = jax.jit(lambda x: jax.jvp(op, (x,), (jnp.ones_like(x),)))(x)
        tangent.block_until_ready()
    np.testing.assert_array_equal(tangent, 2 * np.arange(5))
    assert calls == ["jvp"]
    for transform, expected_call in ((jax.jacfwd, "jvp"), (jax.jacrev, "vjp")):
        calls.clear()
        actual = jax.jit(transform(op))(x)
        np.testing.assert_array_equal(actual, np.diag(2 * np.arange(5)))
        assert calls and set(calls) == {expected_call}


def test_vjp_only_external_adjoint_supports_reverse_mode(gpu):
    adjoint = jb.bind(lambda x, c: 2 * x * c)
    op = jb.bind(gpu.square, vjp=lambda p, c: (adjoint(p[0], c),))
    x = jnp.arange(5.0)
    with jax.transfer_guard_device_to_host("disallow"):
        gradient = jax.jit(jax.grad(lambda x: op(x).sum()))(x).block_until_ready()
    np.testing.assert_array_equal(gradient, 2 * np.arange(5))
    with pytest.raises(TypeError, match="Cannot infer a JVP.*supply jvp"):
        jax.jit(lambda x: jax.jvp(op, (x,), (jnp.ones_like(x),)))(x)


@pytest.mark.parametrize("rules", ["vjp", "both"])
def test_recursive_external_derivative_rules_support_higher_orders(gpu, rules):
    options = {"vjp": lambda p, c: (exponential(p[0]) * c,)}
    if rules == "both":
        options["jvp"] = lambda p, t: exponential(p[0]) * t[0]
    exponential = jb.bind(gpu.exp, out=True, **options)
    x = jnp.array([-0.5, 0.2, 1.0])
    first = jax.grad(lambda x: exponential(x).sum())
    for derivative in (first, jax.jacfwd(first), jax.jacrev(jax.jacfwd(first))):
        with jax.transfer_guard_device_to_host("disallow"):
            actual = jax.jit(derivative)(x).block_until_ready()
        expected = np.zeros(actual.shape)
        for i in range(3):
            expected[(i,) * actual.ndim] = np.exp(float(x[i]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rules", ["vjp", "both"])
def test_multiple_inputs_output_tree_static_keywords_and_nested_batching(rules):
    def forbidden(*args, **kwargs):
        raise AssertionError("Derivative calculation must eliminate the forward call")

    def output(x, y, *, scale):
        return {
            "product": jax.ShapeDtypeStruct(x.shape, x.dtype),
            "sum": jax.ShapeDtypeStruct((), x.dtype),
        }

    def push(p, t, *, scale):
        x, y = p
        dx, dy = t
        return {"product": scale * (dx * y + x * dy), "sum": dx.sum() + dy}

    def pull(p, c, *, scale):
        x, y = p
        return scale * y * c["product"] + c["sum"], (
            scale * x * c["product"]
        ).sum() + c["sum"]

    op = jb.bind(
        forbidden, output, vjp=pull, **({"jvp": push} if rules == "both" else {})
    )

    def loss(x, y):
        result = op(x, y, scale=3.0)
        return result["product"].sum() + 2 * result["sum"]

    x = jnp.array([1.0, -2.0, 0.3])
    y = jnp.array(0.5)
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)
    np.testing.assert_allclose(gradients[0], np.full(3, 3 * 0.5 + 2))
    np.testing.assert_allclose(gradients[1], 3 * np.asarray(x).sum() + 2)
    mixed = jax.jit(jax.jacfwd(jax.grad(loss, argnums=0), argnums=1))(x, y)
    np.testing.assert_array_equal(mixed, np.full(3, 3.0))
    grad = jax.grad(loss, argnums=0)
    batch = jnp.arange(24.0).reshape(2, 4, 3)
    actual = jax.jit(jax.vmap(jax.vmap(grad, in_axes=(0, None)), in_axes=(0, None)))(
        batch, y
    )
    np.testing.assert_array_equal(actual, np.full((2, 4, 3), 3.5))


def test_none_cotangents_and_integer_inputs():
    op = jb.bind(lambda *args: None, vjp=lambda p, c: (2 * p[0] * c, None))
    index = jnp.arange(3, dtype=jnp.int32)
    x = jnp.array([0.5, -2.0, 3.0])

    def f(x):
        return op(x, index)

    gradient = jax.jit(jax.grad(lambda x: f(x).sum()))(x)
    np.testing.assert_array_equal(gradient, 2 * np.asarray(x))
    tangent = jax.jit(lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))[1])(x)
    np.testing.assert_array_equal(tangent, 2 * np.asarray(x))


def test_integer_outputs_have_float0_tangents():
    op = jb.bind(
        lambda x: None,
        output=lambda x: {
            "value": jax.ShapeDtypeStruct(x.shape, x.dtype),
            "label": jax.ShapeDtypeStruct(x.shape, np.int32),
        },
        vjp=lambda p, c: (2 * p[0] * c["value"],),
    )
    x = jnp.array([0.5, -2.0, 3.0])
    result = jax.jit(lambda x: jax.jvp(op, (x,), (jnp.ones_like(x),))[1])(x)
    np.testing.assert_array_equal(result["value"], 2 * np.asarray(x))
    assert result["label"].dtype == jax.dtypes.float0
    assert result["label"].shape == x.shape


@pytest.mark.parametrize("rules", ["vjp", "both"])
def test_derivative_compilation_does_not_materialize_a_dense_jacobian(rules):
    options = {"vjp": lambda p, c: (2 * p[0] * c,)}
    if rules == "both":
        options["jvp"] = lambda p, t: 2 * p[0] * t[0]
    op = jb.bind(lambda x: None, **options)
    spec = jax.ShapeDtypeStruct((1 << 20,), np.float32)
    compiled = jax.jit(jax.grad(lambda x: op(x).sum())).lower(spec).compile()
    memory = compiled.memory_analysis()
    assert memory.output_size_in_bytes == (1 << 20) * 4
    assert memory.temp_size_in_bytes <= memory.output_size_in_bytes


def test_complex_vjp_obeys_jax_bilinear_convention():
    z = jnp.array([1 + 2j, -0.5 + 0.2j])
    op = jb.bind(lambda z: None, vjp=lambda p, c: (3 * p[0] ** 2 * c,))
    expected = np.diag(3 * np.asarray(z) ** 2)
    for transform in (jax.jacfwd, jax.jacrev):
        actual = jax.jit(transform(op, holomorphic=True))(z)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
    norm = jb.bind(
        lambda z: None,
        output=lambda z: jax.ShapeDtypeStruct(z.shape, np.float64),
        vjp=lambda p, c: (2 * jnp.conj(p[0]) * c,),
    )
    gradient = jax.jit(jax.grad(lambda z: norm(z).sum()))(z)
    np.testing.assert_array_equal(gradient, 2 * np.conj(np.asarray(z)))
    tangent = jax.jit(lambda z: jax.jvp(norm, (z,), (jnp.ones_like(z),))[1])(z)
    np.testing.assert_array_equal(tangent, 2 * np.real(np.asarray(z)))


@pytest.mark.parametrize(
    "vjp,error",
    [
        (42, "callable"),
        (lambda p, c: c, "one cotangent"),
        (lambda p, c: (c, c), "one cotangent"),
        (lambda p, c: (jnp.zeros((4,), dtype=p[0].dtype),), "shape/dtype"),
        (lambda p, c: (jnp.zeros(p[0].shape, dtype=jnp.int32),), "shape/dtype"),
    ],
)
def test_invalid_vjp_rules_have_clear_errors(vjp, error):
    with pytest.raises((TypeError, ValueError), match=error):
        op = jb.bind(lambda x: None, vjp=vjp)
        jax.eval_shape(
            jax.grad(lambda x: op(x).sum()), jax.ShapeDtypeStruct((3,), np.float64)
        )


def test_advanced_callback_vjp_uses_flat_output_cotangents():
    def forbidden(*args):
        raise AssertionError("Derivative-only evaluation must eliminate the callback")

    def abstract(x, y):
        return ((x.shape, x.dtype), (x.shape, x.dtype))

    op = jb.get_call(
        forbidden,
        abstract,
        vjp=lambda p, c: (p[1] * c[0] + c[1], p[0] * c[0] + c[1]),
    )

    def loss(x, y):
        product, total = op(x, y)
        return product.sum() + 2 * total.sum()

    x, y = jnp.arange(4.0), jnp.arange(4.0) + 3
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)
    np.testing.assert_array_equal(gradients[0], np.arange(4) + 5)
    np.testing.assert_array_equal(gradients[1], np.arange(4) + 2)


def test_vjp_with_donation_preserves_primal_values_for_reverse_mode(gpu):
    square = jb.bind(
        gpu.square,
        out=True,
        input_output_aliases={0: 0},
        vjp=lambda p, c: (2 * p[0] * c,),
    )
    host = np.arange(8.0)
    x = jnp.asarray(host)
    with jax.transfer_guard_device_to_host("disallow"):
        value, gradient = jax.jit(
            jax.value_and_grad(lambda x: (square(x) ** 2).sum()), donate_argnums=(0,)
        )(x)
        jax.block_until_ready((value, gradient))
    np.testing.assert_array_equal(value, (host**4).sum())
    np.testing.assert_array_equal(gradient, 4 * host**3)


def test_eager_jvp_reuses_derivative_compilation(gpu):
    traces = []

    def rule(p, c):
        traces.append(1)
        return (2 * p[0] * c,)

    op = jb.bind(gpu.square, vjp=rule)
    x = jnp.arange(4.0)
    direction = jnp.ones_like(x)
    for _ in range(3):
        value, tangent = jax.jvp(op, (x,), (direction,))
        jax.block_until_ready((value, tangent))
        np.testing.assert_array_equal(tangent, 2 * np.arange(4))
    assert len(traces) == 1


@pytest.mark.parametrize("rules", ["vjp", "both"])
def test_reusable_linearization_transpose_and_checkpoint(gpu, rules):
    options = {"vjp": lambda p, c: (3 * p[0] ** 2 * c,)}
    if rules == "both":
        options["jvp"] = lambda p, t: 3 * p[0] ** 2 * t[0]
    cube = jb.bind(lambda x: x**3, **options)
    host = np.array([-1.5, 0.2, 2.0])
    x = jnp.asarray(host)
    direction = jnp.array([0.2, -0.3, 0.7])
    @jax.jit
    def calculation(x, direction):
        value, push = jax.linearize(cube, x)
        tangent = push(direction)
        cotangent = jax.linear_transpose(push, direction)(direction)[0]
        gradient = jax.grad(lambda x: (jax.checkpoint(cube)(x) ** 2).sum())(x)
        return value, tangent, cotangent, gradient

    with jax.transfer_guard_device_to_host("disallow"):
        value, tangent, cotangent, gradient = calculation(x, direction)
        jax.block_until_ready((value, tangent, cotangent, gradient))
    np.testing.assert_allclose(value, host**3, rtol=1e-12)
    for actual in (tangent, cotangent):
        np.testing.assert_allclose(
            actual, 3 * host**2 * np.asarray(direction), rtol=1e-12
        )
    np.testing.assert_allclose(gradient, 6 * host**5, rtol=1e-12)
