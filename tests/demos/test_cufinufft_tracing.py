import importlib.util
import jax
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


from examples.cufinufft_demo import nufft1, nufft2


@pytest.mark.parametrize(
    "points,data,error",
    [
        ([((4,), np.float32)], ((4,), np.complex128), "float64"),
        ([((4,), np.float64)], ((4,), np.float64), "complex64 or complex128"),
        ([((4,), np.float64)], ((5,), np.complex128), "point count"),
        ([((4, 1), np.float64)], ((4,), np.complex128), "one-dimensional"),
        ([((4,), np.float64), ((5,), np.float64)], ((4,), np.complex128), "identical"),
        ([((0,), np.float64)], ((0,), np.complex128), "At least one"),
    ],
)
def test_cufinufft_input_validation(points, data, error):
    coords = tuple(jax.ShapeDtypeStruct(*p) for p in points)
    c = jax.ShapeDtypeStruct(*data)
    with pytest.raises((TypeError, ValueError), match=error):
        jax.eval_shape(
            lambda *args: nufft1(args[:-1], args[-1], (5,) * len(coords)), *coords, c
        )


def test_cufinufft_multiple_leading_batch_shapes():
    coords = (jax.ShapeDtypeStruct((11,), np.float64),) * 2
    strengths = jax.ShapeDtypeStruct((2, 3, 11), np.complex128)
    grid = jax.eval_shape(lambda c: nufft1(coords, c, (4, 5)), strengths)
    assert grid.shape == (2, 3, 4, 5)
    values = jax.eval_shape(lambda f: nufft2(coords, f), grid)
    assert values.shape == (2, 3, 11)
