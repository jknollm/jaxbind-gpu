import jax
import jax.numpy as jnp
import numpy as np

import jaxbind_gpu as jb


def test_explicit_effect_survives_unused_outputs(gpu):
    calls = []

    def effect(out, args, kwargs_dump):
        calls.append(1)

    op = jb.get_call(effect, lambda x: (), has_side_effect=True)

    @jax.jit
    def execute(x):
        op(x)
        op(x)
        return x + 1

    np.testing.assert_array_equal(execute(jnp.arange(4.0)), np.arange(4.0) + 1)
    jax.effects_barrier()
    assert calls == [1, 1]


def test_explicit_alias_and_donation(gpu):
    observed = []

    def scale(out, args, kwargs_dump):
        observed.append(out[0].data.ptr == args[0].data.ptr)
        gpu.multiply(args[0], 3.0, out=out[0])

    def abstract(x):
        return ((x.shape, x.dtype),)

    op = jb.get_call(scale, abstract, input_output_aliases={0: 0})
    x = jnp.arange(8.0, dtype=jnp.float32)
    result = jax.jit(op, donate_argnums=(0,))(x)[0]
    np.testing.assert_array_equal(result, np.arange(8.0) * 3.0)
    assert observed == [True]
