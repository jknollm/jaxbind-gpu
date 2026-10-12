import jax
import jax.numpy as jnp
import numpy as np

import jaxbind_gpu as jb


def test_fft_resource_lifetime_batching_and_complex_transpose(gpu):
    from cupyx.scipy.fft import fftn, get_fft_plan

    def execute(out, args, kwargs_dump):
        context = jb.current_context()
        x = args[0]
        plan = context.cached(
            (x.shape, x.dtype.str), lambda: get_fft_plan(x, axes=(-2, -1))
        )
        temporary = fftn(x, axes=(-2, -1), norm="ortho", plan=plan)
        gpu.copyto(out[0], temporary)
        context.retain(temporary)

    def abstract(x):
        return ((x.shape, x.dtype),)

    def batch(call, args, axes):
        return call(jnp.moveaxis(args[0], axes[0], 0)), (0,)

    op = jb.get_linear_call(execute, execute, abstract, abstract, batching_rule=batch)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(3, 8, 7)) + 1j * rng.normal(size=(3, 8, 7)))
    actual = jax.jit(jax.vmap(op))(x)[0]
    expected = np.fft.fftn(np.asarray(x), axes=(-2, -1), norm="ortho")
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    weights = jnp.conj(x)
    _, pullback = jax.vjp(lambda x: op(x)[0], x)
    actual = jax.jit(pullback)(weights)[0]
    np.testing.assert_allclose(
        actual,
        np.fft.fftn(np.asarray(weights), axes=(-2, -1), norm="ortho"),
        atol=1e-12,
    )
