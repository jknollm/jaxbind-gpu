"""Measure cuFINUFFT wrapper memory in one fresh process per case and variant.

Library-private allocations are visible only to the sampled driver counter.
The sampler is a lower bound on transient usage, not an allocator audit.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "released_dynamic",
    "cached_dynamic",
    "cached_prepared",
    "bound_dynamic",
    "bound_prepared",
)


def worker(args, case):
    import cupy as cp
    import jax
    import jax.numpy as jnp
    import numpy as np
    from benchmark_common import check_exclusive

    sys.path.insert(0, str(ROOT))
    reference_path = (
        ROOT
        / ".references"
        / ("cached-reference" if args.variant.startswith("cached") else "gpu-reference")
    )
    sys.path.insert(0, str(reference_path))
    import jax_finufft as reference
    from jax_finufft import jax_finufft_gpu
    from jax_finufft.options import Opts
    from examples import cufinufft_demo as ours
    import jaxbind_gpu as jb

    assert Path(jax_finufft_gpu.__file__).is_relative_to(reference_path)
    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0]
    gpu = check_exclusive(cp)
    shape = tuple(case["grid"])
    batch, kind = case["batch"], case["kind"]
    dtype = np.dtype(case["dtype"])
    real = np.float32 if dtype == np.complex64 else np.float64
    rng = np.random.default_rng(512 + args.case)
    points = tuple(
        jnp.asarray(rng.uniform(-np.pi, np.pi, case["points"]).astype(real))
        for _ in shape
    )
    other_points = tuple(p + real(0.013) for p in points)
    data_shape = (case["points"],) if kind == 1 else shape
    if batch > 1:
        data_shape = (batch,) + data_shape
    data = jnp.asarray(
        (
            (rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape))
            / np.sqrt(np.prod(data_shape))
        ).astype(dtype)
    )
    other_data = data * real(1.1)
    jax.block_until_ready((points, other_points, data, other_data))
    opts = Opts(gpu_upsampfac=case["upsampfac"], gpu_method=case["gpu_method"])
    sign = 1 if kind == 1 else -1
    options = dict(
        isign=sign,
        eps=case["eps"],
        upsampfac=case["upsampfac"],
        gpu_method=case["gpu_method"],
    )

    def external(value, *coordinates):
        coordinates = tuple(p[None, :] if batch > 1 else p for p in coordinates)
        if kind == 1:
            return reference.nufft1(
                shape, value, *coordinates, iflag=sign, eps=case["eps"], opts=opts
            )
        return reference.nufft2(
            value, *coordinates, iflag=sign, eps=case["eps"], opts=opts
        )

    prepared = args.variant.endswith("prepared")
    mode = None
    if args.variant.startswith("cached"):
        library = ctypes.CDLL(jax_finufft_gpu.__file__)
        mode = library.jax_finufft_benchmark_mode
        mode.argtypes = [ctypes.c_int]
        mode.restype = None
        mode(int(prepared))
    if args.variant.startswith("bound"):
        if prepared:
            prepare = ours.prepare_nufft1 if kind == 1 else ours.prepare_nufft2
            function = prepare(points, *((shape,) if kind == 1 else ()), **options)
        else:

            def function(value, *coordinates):
                if kind == 1:
                    return ours.nufft1(coordinates, value, shape, **options)
                return ours.nufft2(coordinates, value, **options)
    else:
        function = external
    coordinates_b = points if prepared else other_points
    sets = [(data, *points), (other_data, *coordinates_b)]
    call_sets = (
        [(data,), (other_data,)]
        if prepared and args.variant.startswith("bound")
        else sets
    )
    compiled = jax.jit(function).lower(*call_sets[0]).compile()
    cp.cuda.Device().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    before_warm_driver = total - free
    with jax.transfer_guard_device_to_host("disallow"):
        for i in range(4):
            output = compiled(*call_sets[i % 2])
            output.block_until_ready()
            jb.synchronize()
            del output
    cp.cuda.Device().synchronize()
    gc.collect()
    initial_jax = device.memory_stats()
    free, total = cp.cuda.runtime.memGetInfo()
    initial_driver = total - free
    peak = [initial_driver]
    stop = threading.Event()

    def sample_driver():
        with cp.cuda.Device(0):
            while not stop.wait(0.001):
                free, total = cp.cuda.runtime.memGetInfo()
                peak[0] = max(peak[0], total - free)

    sampler = threading.Thread(target=sample_driver, daemon=True)
    sampler.start()
    try:
        with jax.transfer_guard_device_to_host("disallow"):
            outputs = [compiled(*call_sets[i % 2]) for i in range(args.count)]
            jax.block_until_ready(outputs)
            jb.synchronize()
        cp.cuda.Device().synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        peak[0] = max(peak[0], total - free)
        final_jax = device.memory_stats()
        runtime = jb.runtime_stats()
        pool = cp.get_default_memory_pool()
        cupy_active, cupy_reserved = pool.used_bytes(), pool.total_bytes()
        check_exclusive(cp)
    finally:
        stop.set()
        sampler.join()
    # Reference execution follows measurement, so its plans and outputs do not
    # contaminate the variant's reported memory. Validate both resident sets.
    if mode is not None:
        mode(0)
    oracle = jax.jit(external)
    tolerance = 5e-5 if dtype == np.complex64 else 2e-9
    for i, output in enumerate(outputs):
        expected = np.asarray(oracle(*sets[i % 2]))
        np.testing.assert_allclose(
            np.asarray(output), expected, atol=tolerance, rtol=tolerance
        )
    analysis = compiled.memory_analysis()
    result = dict(
        case={k: v for k, v in case.items() if k != "timings"},
        variant=args.variant,
        count=args.count,
        gpu=gpu,
        package_version=jb.__version__,
        reference_version=reference.__version__,
        reference_extension_sha256=hashlib.sha256(
            Path(jax_finufft_gpu.__file__).read_bytes()
        ).hexdigest(),
        driver_before_plan_warmup_bytes=before_warm_driver,
        driver_initial_used_bytes=initial_driver,
        driver_sampled_peak_used_bytes=peak[0],
        jax_initial_memory_stats=initial_jax,
        jax_final_memory_stats=final_jax,
        cupy_final_active_bytes=cupy_active,
        cupy_final_reserved_bytes=cupy_reserved,
        executable_bytes={
            k: getattr(analysis, k)
            for k in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        output_bytes_per_call=outputs[0].nbytes,
        runtime=runtime,
        correctness_checked=True,
    )
    del outputs
    ours.synchronize(clear_cache=True)
    if mode is not None:
        mode(0)
    cp.cuda.Device().synchronize()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/current-wrapper-memory.json",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    timing_path = ROOT / "benchmarks/results/current-cached-wrapper.json"
    timing = json.loads(timing_path.read_text())
    if args.variant:
        print(json.dumps(worker(args, timing["results"][args.case])))
        return
    rows = []
    payload = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        package_version=timing["execution_package"]["version"],
        methodology="One fresh process per configuration and variant, identical two resident input sets, four warmed calls and four fresh retained outputs. Prepared geometry is fixed while data changes. Sampling at 1 ms yields a lower bound on driver peak and includes private cuFINUFFT allocations. JAX live/pool peaks and compiler scratch remain separate. Correctness oracles run after measurement. GPU exclusivity checked before and after each worker.",
        source_sha256={
            str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            **timing["source_sha256"],
        },
        execution_package=timing["execution_package"],
        packages=timing["packages"],
        timing_evidence_sha256=hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        results=rows,
    )
    for index in (
        args.cases if args.cases is not None else range(len(timing["results"]))
    ):
        for variant in VARIANTS:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_gpu.py"),
                    "--wait",
                    "60",
                    str(Path(__file__).resolve()),
                    "--case",
                    str(index),
                    "--variant",
                    variant,
                    "--count",
                    str(args.count),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                print(result.stdout, end="", flush=True)
                print(result.stderr, end="", file=sys.stderr, flush=True)
                result.check_returncode()
            row = json.loads(result.stdout.splitlines()[-1])
            rows.append(row)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"{len(rows)}: case {index}, {variant}, driver peak {row['driver_sampled_peak_used_bytes'] / 2**20:.3f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
