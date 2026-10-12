"""Compare unchanged return-only CuPy functions with both output policies.

Timing variants use changing resident inputs and fresh, held outputs. Memory
variants run in fresh processes, instrument the fallback CuPy allocator, and
exclude numerical-oracle allocations. Use run_gpu.py to select an idle GPU.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from benchmark_common import DirectContext, check_exclusive, sample


def problem(name, size, cp, np, direct_context):
    rng = np.random.default_rng(42)
    sets = [rng.normal(size=size).astype(np.float32) for _ in range(2)]
    if name == "square":
        return cp.square, sets, np.square
    if name == "sort":
        return cp.sort, sets, np.sort
    if name == "scan":
        sets = [np.ones(size, np.float32), np.full(size, 2, np.float32)]
        return cp.cumsum, sets, np.cumsum
    if name == "fft":
        from cupyx.scipy.fft import fft, get_fft_plan
        import jaxbind_gpu as jb

        def external(x):
            try:
                context = jb.current_context()
            except RuntimeError:
                context = direct_context
            plan = context.cached((x.shape, x.dtype.str), lambda: get_fft_plan(x))
            return fft(x, plan=plan)

        sets = [x.astype(np.complex64) for x in sets]
        return external, sets, lambda x: np.fft.fft(x.astype(np.complex128))
    if name == "gaussian_filter":
        from cupyx.scipy.ndimage import gaussian_filter
        from scipy.ndimage import gaussian_filter as reference

        return lambda x: gaussian_filter(x, 1.0), sets, lambda x: reference(x, 1.0)
    if name == "cholesky":
        # Size is the number of matrix elements, keeping benchmark storage small.
        n = int(np.sqrt(size))
        sets = []
        for _ in range(2):
            a = rng.normal(size=(n, n)).astype(np.float32)
            sets.append(a @ a.T + np.eye(n, dtype=np.float32) * n)
        return cp.linalg.cholesky, sets, np.linalg.cholesky
    raise ValueError(name)


def run(args):
    import cupy as cp
    import jax
    import numpy as np
    import jaxbind_gpu as jb

    device = jax.devices("gpu")[0]
    gpu = check_exclusive(cp)
    context = DirectContext(cp)
    function, hosts, oracle = problem(args.app, args.size, cp, np, context)
    expected = [oracle(x) for x in hosts]
    spec = jax.ShapeDtypeStruct(expected[0].shape, hosts[0].dtype)
    sets = [(jax.device_put(x, device),) for x in hosts]
    jax.block_until_ready(sets)
    cupy_sets = [tuple(cp.from_dlpack(x) for x in a) for a in sets]

    def direct(x):
        with context.stream:
            return function(x)

    pointers = {"copy": [], "allocate": []}

    def observed(mode):
        def external(x):
            value = function(x)
            if pointers is not None:
                pointers[mode].append(value.data.ptr)
            return value

        return external

    variants = {
        "native": (direct, cupy_sets, lambda results: context.synchronize()),
    }
    memory = {}
    for mode in ("copy", "allocate"):
        compiled = (
            jax.jit(
                jb.bind(
                    observed(mode),
                    output=spec,
                    out=False if mode == "copy" else "allocate",
                )
            )
            .lower(*sets[0])
            .compile()
        )
        analysis = compiled.memory_analysis()
        memory[mode] = {
            key: getattr(analysis, key)
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
        variants[mode] = compiled, sets, jax.block_until_ready

    audits = {}
    for name, (operation, argument_sets, synchronize) in variants.items():
        if args.memory_variant and name != args.memory_variant:
            continue
        matches = []
        for i, a in enumerate(argument_sets):
            value = operation(*a)
            synchronize([value])
            if name != "native":
                matches.append(pointers[name][-1] == value.unsafe_buffer_pointer())
            actual = cp.asnumpy(value) if name == "native" else np.asarray(value)
            # FFT cancellation needs an absolute tolerance that follows the
            # sqrt(N) spectral scale, rather than a relative test near zero.
            atol = (
                16 * np.finfo(np.float32).eps * np.sqrt(args.size)
                if args.app == "fft"
                else 2e-4
            )
            np.testing.assert_allclose(actual, expected[i], rtol=2e-5, atol=atol)
            del value
        if name != "native":
            audits[name] = matches
        for _ in range(4):
            sample(
                operation, argument_sets, synchronize, count=args.count, latency=False
            )
    # Stop recording pointers before measuring either time or memory.
    pointers = None
    if args.memory_variant:
        operation, argument_sets, synchronize = variants[args.memory_variant]
        # Keep warmed plans on every path. Clearing only the binding's cache
        # would count fresh plan creation against it but not against native.
        jb.synchronize()
        cp.cuda.Device().synchronize()
        gc.collect()
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        initial = pool.used_bytes()
        peak = [initial]
        allocator = cp.cuda.get_allocator()

        def observe_allocator(size):
            value = allocator(size)
            peak[0] = max(peak[0], pool.used_bytes())
            return value

        jax_initial = device.memory_stats()
        cp.cuda.set_allocator(observe_allocator)
        try:
            outputs = [operation(*argument_sets[i % 2]) for i in range(args.count)]
            synchronize(outputs)
            jb.synchronize()
            result = {
                "variant": args.memory_variant,
                "cupy_initial_active_bytes": initial,
                "cupy_peak_additional_active_bytes": peak[0] - initial,
                "jax_initial_memory_stats": jax_initial,
                "jax_final_memory_stats": device.memory_stats(),
                "output_bytes_per_call": int(np.prod(spec.shape))
                * np.dtype(spec.dtype).itemsize,
                "count": args.count,
                "executable_bytes": memory,
            }
        finally:
            cp.cuda.set_allocator(allocator)
    else:
        values = {name: {"latency": [], "throughput": []} for name in variants}
        with jax.transfer_guard_device_to_host("disallow"):
            for repetition in range(args.repeats):
                check_exclusive(cp)
                names = list(variants)
                names = names[repetition % 3 :] + names[: repetition % 3]
                if repetition % 2:
                    names.reverse()
                for name in names:
                    operation, argument_sets, synchronize = variants[name]
                    for metric in ("latency", "throughput"):
                        values[name][metric].append(
                            sample(
                                operation,
                                argument_sets,
                                synchronize,
                                count=args.count,
                                latency=metric == "latency",
                            )
                        )
                check_exclusive(cp)
        result = {
            "timings": {
                name: {
                    metric + "_us": statistics.median(samples)
                    for metric, samples in metrics.items()
                }
                | {"samples_us": metrics}
                for name, metrics in values.items()
            },
            "paired_saved_us": {
                metric: statistics.median(
                    a - b
                    for a, b in zip(values["copy"][metric], values["allocate"][metric])
                )
                for metric in ("latency", "throughput")
            },
            "result_pointer_is_xla_output": audits,
            "executable_bytes": memory,
        }
    check_exclusive(cp)
    context.close()
    jb.synchronize(clear_cache=True)
    package = Path(jb.__file__).parent
    return result | {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app": args.app,
        "size": args.size,
        "gpu": gpu,
        "package_version": jb.__version__,
        "packages": {
            "jax": jax.__version__,
            "cupy": cp.__version__,
            "numpy": np.__version__,
        },
        "resource_contract": "FFT uses identical explicit cuFFT plans on all paths. Persistent plans are created through the execution context cache, outside output forwarding. Other functions use their ordinary return-only interface.",
        "count": args.count,
        "repeats": args.repeats,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (*sorted(package.glob("*.py")), *sorted(package.glob("*.so")))
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_support_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_common.py").read_bytes()
        ).hexdigest(),
        "benchmark_helpers_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_applications.py").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apps",
        nargs="+",
        default=["square", "sort", "scan", "fft", "gaussian_filter", "cholesky"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576])
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--app")
    parser.add_argument("--size", type=int)
    parser.add_argument("--memory-variant", choices=("native", "copy", "allocate"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/current-output-allocation.json"),
    )
    args = parser.parse_args()
    if min(*args.sizes, args.count, args.repeats) < 1:
        parser.error("sizes, count, and repeats must be positive")
    if args.app:
        print(json.dumps(run(args)), flush=True)
        return
    payload = {
        "methodology": "Identical return-only CuPy functions for native, default-copy, and allocation-forwarding variants. Two changing resident input sets; fresh held outputs. Rotating variant order, immediate untimed warm batch before each timed batch. Independent NumPy/SciPy oracle checks outside timing. Opt-in forwarding changes allocation policy only. Memory workers are separate processes per application, size, and variant; CuPy peaks omit XLA outputs and JAX peaks include them.",
        "results": [],
    }
    for app in args.apps:
        for size in args.sizes:
            for variant in ("native", "copy", "allocate") if args.memory else (None,):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--app",
                    app,
                    "--size",
                    str(size),
                    "--count",
                    str(args.count),
                    "--repeats",
                    str(args.repeats),
                ]
                if variant:
                    command.extend(["--memory-variant", variant])
                child = subprocess.run(command, text=True, capture_output=True)
                if child.returncode:
                    raise RuntimeError(
                        f"Benchmark worker failed: {command}\n{child.stderr}"
                    )
                row = json.loads(child.stdout)
                payload["results"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "app": app,
                            "size": size,
                            "variant": variant,
                            "saved_us": row.get("paired_saved_us"),
                            "cupy_peak": row.get("cupy_peak_additional_active_bytes"),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
