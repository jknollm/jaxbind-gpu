"""Warp's library routine must preserve values, derivatives and GPU ownership."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("graph", [False, True])
def test_warp_sparse_transforms_and_borrowed_buffers(gpu, graph, monkeypatch):
    wp = pytest.importorskip("warp")
    from warp import sparse
    from examples.warp_sparse import make_matvec
    from jaxbind_gpu import advanced

    wp.init()
    # A rectangular, nonsymmetric matrix makes the reverse rule observable.
    dense = np.array([[2, -1, 3], [4, 0.5, -2]], dtype="float32")
    rows, columns = np.indices(dense.shape)
    matrix = sparse.bsr_from_triplets(
        2,
        3,
        wp.array(rows.ravel().astype("int32"), device="cuda:0"),
        wp.array(columns.ravel().astype("int32"), device="cuda:0"),
        wp.array(dense.ravel(), device="cuda:0"),
    )
    wp.synchronize_device(matrix.device)
    observed = []
    original = wp.from_dlpack

    def borrow(array, *args, **kwargs):
        view = original(array, *args, **kwargs)
        assert view.ptr == array.data.ptr
        observed.append(view.ptr)
        return view

    monkeypatch.setattr(wp, "from_dlpack", borrow)
    op = make_matvec(matrix, graph_compatible=graph)
    x_host = np.array([1, -2, 0.5], dtype="float32")
    x = jnp.asarray(x_host)
    # Compile Warp kernels outside CUDA capture.
    op(x).block_until_ready()
    options = (
        {
            "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
            "xla_gpu_graph_min_graph_size": 1,
        }
        if graph
        else None
    )
    function = jax.jit(op, compiler_options=options)
    observed.clear()
    before = advanced.runtime_stats()["graph_captures"]
    with jax.transfer_guard_device_to_host("disallow"):
        value = function(x).block_until_ready()
    assert observed[:2] == [x.unsafe_buffer_pointer(), value.unsafe_buffer_pointer()]
    np.testing.assert_allclose(value, dense @ x_host, rtol=2e-6, atol=2e-6)
    direction = np.array([0.3, -0.4, 0.7], dtype="float32")
    weights = np.array([-0.2, 0.8], dtype="float32")
    with jax.transfer_guard_device_to_host("disallow"):
        tangent = jax.jit(lambda x, t: jax.jvp(op, (x,), (t,))[1])(
            x, jnp.asarray(direction)
        )
        cotangent = jax.jit(lambda x, w: jax.vjp(op, x)[1](w)[0])(
            x, jnp.asarray(weights)
        )
        hessian = jax.jit(jax.hessian(lambda z: jnp.sum(op(z) ** 2)))(x)
        batch = jnp.stack([x, 2 * x, -x])
        mapped = jax.jit(jax.vmap(op), compiler_options=options)(batch)
        outputs = [function(jnp.asarray(scale * x_host)) for scale in range(1, 13)]
        jax.block_until_ready((tangent, cotangent, hessian, mapped, outputs))
    np.testing.assert_allclose(tangent, dense @ direction, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(cotangent, dense.T @ weights, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(hessian, 2 * dense.T @ dense, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(mapped, np.array([1, 2, -1])[:, None] * (dense @ x_host))
    for scale, output in enumerate(outputs, 1):
        np.testing.assert_allclose(
            output, scale * (dense @ x_host), rtol=2e-6, atol=2e-6
        )
    if graph:
        assert advanced.runtime_stats()["graph_captures"] > before
    advanced.synchronize(clear_cache=True)
