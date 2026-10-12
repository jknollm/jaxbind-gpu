import jax
import numpy as np
import pytest

import jaxbind_gpu as jb
from jaxbind_gpu import advanced


def test_shape_evaluation_does_not_execute_callback():
    def forbidden(out, args, kwargs_dump):
        raise AssertionError("Shape tracing must not invoke external operations")

    def abstract(x, **kwargs):
        return ((x.shape, x.dtype),)

    op = advanced.get_linear_call(forbidden, forbidden, abstract, abstract)
    result = jax.eval_shape(op, jax.ShapeDtypeStruct((3, 7), np.float32))
    assert result[0].shape == (3, 7)
    assert result[0].dtype == np.float32


def test_cpu_execution_has_clear_error():
    if jax.default_backend() != "cpu":
        pytest.skip("Run this check with JAX_PLATFORMS=cpu")

    def abstract(x, **kwargs):
        return ((x.shape, x.dtype),)

    op = jb.get_linear_call(lambda *args: None, lambda *args: None, abstract, abstract)
    with pytest.raises(NotImplementedError, match="require a CUDA device"):
        jax.jit(op)(np.ones(4))
