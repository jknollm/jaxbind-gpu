"""Reuse library plans and XLA scratch during CUDA graph replay."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cupyx.scipy.fft import fftn, get_fft_plan

from jaxbind_gpu import bind
from jaxbind_gpu import advanced as jb


def transform(x, *, out):
    context = jb.current_context()
    # The factory uses shapes only. It allocates persistent plan storage outside
    # the temporary workspace and never stores borrowed input/output arrays.
    plan = context.cached(
        (x.shape, x.dtype.str),
        lambda: get_fft_plan(cp.empty(x.shape, x.dtype)),
    )
    value = fftn(x, plan=plan)
    cp.copyto(out, value)
    context.retain(value)  # The output copy still uses this temporary allocation.


def main():
    fft = bind(
        transform,
        transpose=transform,  # The unnormalized Fourier matrix is symmetric.
        out=True,
        graph_compatible=True,
        workspace_bytes=lambda x: 2 * x.size * np.dtype(x.dtype).itemsize + 1024,
    )
    host = np.random.default_rng(11).normal(size=256).astype("complex64")
    x = jnp.asarray(host)
    jax.jit(fft)(x).block_until_ready()  # Warm kernels before graph capture.
    options = {
        "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
        "xla_gpu_graph_min_graph_size": 1,
    }

    @jax.jit(compiler_options=options)
    def repeated(x):
        return jax.lax.fori_loop(0, 4, lambda i, x: fft(x), x, unroll=True)

    with jax.transfer_guard_device_to_host("disallow"):
        value = repeated(x).block_until_ready()
        changed = repeated(x * 2).block_until_ready()
    expected = host.astype("complex128")
    for _ in range(4):
        expected = np.fft.fftn(expected)
    np.testing.assert_allclose(value, expected, rtol=2e-5, atol=0.03)
    np.testing.assert_allclose(changed, 2 * expected, rtol=2e-5, atol=0.06)
    stats = jb.runtime_stats()
    assert stats["graph_captures"] > 0
    assert stats["workspace_peak_bytes"] > 0
    print("Cached FFT: changing graph inputs and NumPy reference passed")
    jb.synchronize(clear_cache=True)


if __name__ == "__main__":
    main()
