"""Measure warmed end-to-end throughput on one GPU, including host dispatch."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jaxbind_gpu as jb


def measure(function, synchronize, *, count=50, repeats=5):
    for _ in range(5):
        function()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(count):
            function()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / count)
    return dict(
        median_us=statistics.median(samples), min_us=min(samples), samples_us=samples
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/benchmarks.json")
    args = parser.parse_args()
    jax.devices()
    import cupy as cp
    import cufinufft
    from examples import cufinufft_demo
    from examples.cufinufft_demo import nufft1, prepare_nufft1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from native_scale import binding

    native, kernel = binding()

    def scale(out, inputs, config):
        cp.multiply(inputs[0], jb.load_kwargs(config)["factor"], out=out[0])

    def abstract(x, **kwargs):
        return ((x.shape, x.dtype),)

    python = jb.get_linear_call(scale, scale, abstract, abstract)
    results = {}
    sync = cp.cuda.Device().synchronize
    for size in (1024, 1048576):
        x = jnp.ones(size, dtype=jnp.float32)
        cx = cp.ones(size, dtype="float32")
        out = cp.empty_like(cx)
        jnative = jax.jit(lambda x: native(x, factor=1.01)[0])
        jpython = jax.jit(lambda x: python(x, factor=1.01)[0])
        jbuiltin = jax.jit(lambda x: x * jnp.float32(1.01))
        functions = {
            "direct_cupy_kernel": lambda: kernel(
                ((size + 255) // 256,),
                (256,),
                (cx, out, np.int64(size), np.float32(1.01)),
            ),
            "native_cabi": lambda: jnative(x),
            "python_callback": lambda: jpython(x),
            "jax_builtin": lambda: jbuiltin(x),
        }
        for name, function in functions.items():
            results[f"scale/{size}/{name}"] = measure(function, sync)
        for name, op in (("native_cabi", native), ("python_callback", python)):
            loop = jax.jit(
                lambda x: jax.lax.fori_loop(
                    0, 50, lambda i, x: op(x, factor=1.001)[0], x
                )
            )
            timing = measure(lambda: loop(x), sync, count=10)
            timing["median_us_per_operation"] = timing["median_us"] / 50
            results[f"scale_loop/{size}/{name}"] = timing

    rng = np.random.default_rng(83)
    for point_count, modes in ((4096, 4096), (131072, 131072)):
        x = jnp.asarray(rng.uniform(-np.pi, np.pi, point_count).astype("float32"))
        c = jnp.ones(point_count, dtype=jnp.complex64)
        cx, cc = cp.asarray(x), cp.asarray(c)
        out = cp.empty(modes, dtype="complex64")
        plan = cufinufft.Plan(1, modes, dtype="complex64", eps=1e-5)
        plan.setpts(cx)
        operation = jax.jit(lambda x, c: nufft1(x, c, modes, eps=1e-5))
        python_operation = jax.jit(
            lambda x, c: nufft1(x, c, modes, eps=1e-5, backend="python")
        )
        prepared = jax.jit(prepare_nufft1(x, modes, eps=1e-5))
        results[f"nufft1/{point_count}/direct_reused_execute"] = measure(
            lambda: plan.execute(cc, out=out), sync
        )

        def direct_dynamic():
            plan.setpts(cx)
            plan.execute(cc, out=out)

        results[f"nufft1/{point_count}/direct_reused_setpts_execute"] = measure(
            direct_dynamic, sync
        )
        results[f"nufft1/{point_count}/binding_native_dynamic"] = measure(
            lambda: operation(x, c), sync
        )
        results[f"nufft1/{point_count}/binding_python_dynamic"] = measure(
            lambda: python_operation(x, c), sync
        )
        results[f"nufft1/{point_count}/binding_native_prepared"] = measure(
            lambda: prepared(c), sync
        )

        def direct_fresh():
            fresh = cufinufft.Plan(1, modes, dtype="complex64", eps=1e-5)
            fresh.setpts(cx)
            fresh.execute(cc, out=out)
            sync()

        results[f"nufft1/{point_count}/direct_fresh_plan"] = measure(
            direct_fresh, sync, count=10
        )
        plan = None

    sync()
    cufinufft_demo.synchronize(clear_cache=True)
    payload = dict(
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "cupy-cuda12x", "cufinufft", "numpy")
        },
        methodology="Median of five warmed wall-clock batches, device synchronization after each batch; includes host dispatch. Loops contain 50 dependent calls.",
        results=results,
    )
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    for name, result in results.items():
        print(f"{name}: {result['median_us']:.2f} us", flush=True)


if __name__ == "__main__":
    main()
