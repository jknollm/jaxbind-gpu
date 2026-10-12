"""Measure allocation peaks in fresh processes, separately from timing runs.

CuPy allocator peaks include every allocation made through its allocator, even
inside an XLA callback thread. JAX and Torch report their own allocation peaks.
Driver memory is sampled and is a lower bound on process peak VRAM. Reserved
allocator memory is reported separately from live allocations. Correctness
oracle allocations are excluded from the measurement interval.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))


def worker(args):
    if args.worker_variant == "jaxbind_gpu_graph":
        # XLA reads this runtime trace-cache option before backend creation.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_cmd_buffer_trace_cache_size={args.trace_cache_size}"
        )
    if args.torch_path:
        sys.path.insert(0, str(args.torch_path.resolve()))
    import cupy as cp
    import jax
    import numpy as np
    import jaxbind_gpu as jb
    from applications.workloads import build
    from benchmark_common import DirectContext, check_exclusive

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0]
    gpu = check_exclusive(cp)
    workload = build(args.worker_app, args.size)
    graph = args.worker_variant == "jaxbind_gpu_graph"
    cached = args.worker_variant == "jaxbind_gpu_cached"
    allocation_control = args.worker_variant == "jax_allocation_control"
    if args.workspace_bytes and not args.worker_variant.startswith("direct_library"):
        if not hasattr(workload.operator, "workspace_bytes"):
            raise ValueError("The selected native operator manages its own workspace")
        workload.operator.workspace_bytes = args.workspace_bytes
        workload.operator.enable_graph(graph or cached)
    if graph or cached:
        workload.operator.enable_graph()
    if args.iterations > 1:
        spec = jax.eval_shape(
            workload.bound,
            *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in workload.host_args),
        )
        last = workload.host_args[-1]
        if (spec.shape, spec.dtype) != (last.shape, last.dtype):
            raise ValueError("Iterations require matching input/output shape and dtype")

    def bound_program(*a):
        if allocation_control:
            spec = jax.eval_shape(workload.bound, *a)
            if (spec.shape, spec.dtype) != (a[-1].shape, a[-1].dtype):
                raise ValueError(
                    "Allocation control requires matching input/output shape and dtype"
                )
            # A plain JAX kernel with a fresh, equally sized output. This is
            # an allocator control, not an equivalent application algorithm.
            return a[-1] + np.array(1, dtype=a[-1].dtype)
        if args.iterations == 1:
            return workload.bound(*a)
        return jax.lax.fori_loop(
            0,
            args.iterations,
            lambda i, x: workload.bound(*a[:-1], x),
            a[-1],
            unroll=args.unroll,
        )

    sets = [
        tuple(jax.device_put(a, device) for a in s) for s in workload.argument_sets()
    ]
    jax.block_until_ready(sets)
    context = DirectContext(cp)
    cupy_sets = [tuple(cp.from_dlpack(a) for a in s) for s in sets]
    direct = (
        args.worker_variant.startswith("direct_library")
        or args.worker_variant == "native_return"
    )
    compiled = None
    native_graph_storage = None
    if direct:

        def function(*a):
            with context.stream:
                for _ in range(args.iterations):
                    value = workload.direct(a, context)
                    a = (*a[:-1], value)
                return value

        call_sets = cupy_sets

        if args.worker_variant == "native_return":
            if args.iterations != 1:
                raise ValueError("Native return memory baselines require one iteration")
            from applications.native_baselines import returning_baseline

            baseline = returning_baseline(workload, cupy_sets, context)
            if baseline is None:
                raise ValueError(f"No native return baseline for {workload.name}")
            function, call_sets = baseline

        def synchronize(outputs):
            context.synchronize()

        if args.worker_variant == "direct_library_graph":
            native_graph_storage = cp.cuda.MemoryPool()
            captures = []
            for a in cupy_sets:
                function(*a)
                context.synchronize()
                with (
                    context.stream,
                    cp.cuda.using_allocator(native_graph_storage.malloc),
                ):
                    context.stream.begin_capture()
                    value = function(*a)
                    captured = context.stream.end_capture()
                captures.append((captured, value))

            def direct_graph(*a):
                index = 0 if a[-1].data.ptr == cupy_sets[0][-1].data.ptr else 1
                captured, value = captures[index]
                with context.stream:
                    captured.launch(stream=context.stream)
                    return value.copy()

            function = direct_graph
    else:
        options = (
            {
                "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
                "xla_gpu_graph_min_graph_size": 1,
            }
            if graph
            else None
        )
        compiled = (
            jax.jit(bound_program, compiler_options=options).lower(*sets[0]).compile()
        )
        function, call_sets = compiled, sets

        def synchronize(outputs):
            jax.block_until_ready(outputs)
            jb.synchronize()

    # Warm the exact implementation/stream, retain no returned outputs.
    for i in range(4):
        output = function(*call_sets[i % 2])
        synchronize([output])
        del output
    cp.cuda.Device().synchronize()
    gc.collect()
    pool = cp.get_default_memory_pool()
    initial_used, initial_reserved = pool.used_bytes(), pool.total_bytes()
    peaks = {"used": initial_used, "reserved": initial_reserved}
    allocator = cp.cuda.get_allocator()

    def observe_allocator(size):
        allocation = allocator(size)
        peaks["used"] = max(peaks["used"], pool.used_bytes())
        peaks["reserved"] = max(peaks["reserved"], pool.total_bytes())
        return allocation

    torch = sys.modules.get("torch")
    torch_initial = None
    if torch is not None:
        torch.cuda.synchronize()
        torch_initial = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    jax_initial = device.memory_stats()
    warp = sys.modules.get("warp")
    warp_device = (
        workload.operator.matrix.device if workload.name == "warp_sparse" else None
    )
    warp_initial = (
        warp.get_mempool_used_mem_current(warp_device) if warp_device else None
    )
    warp_initial_high = (
        warp.get_mempool_used_mem_high(warp_device) if warp_device else None
    )
    warp_peak = [warp_initial]
    free, total = cp.cuda.runtime.memGetInfo()
    driver_initial = total - free
    driver_peak = [driver_initial]
    stop = threading.Event()

    def sample_driver():
        with cp.cuda.Device(0):
            while not stop.wait(0.001):
                free, total = cp.cuda.runtime.memGetInfo()
                driver_peak[0] = max(driver_peak[0], total - free)
                if warp_device:
                    warp_peak[0] = max(
                        warp_peak[0], warp.get_mempool_used_mem_current(warp_device)
                    )

    sampler = threading.Thread(target=sample_driver, daemon=True)
    sampler.start()
    cp.cuda.set_allocator(observe_allocator)
    try:
        outputs = [function(*call_sets[i % 2]) for i in range(args.count)]
        synchronize(outputs)
        cp.cuda.Device().synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        driver_peak[0] = max(driver_peak[0], total - free)
        check_exclusive(cp)
        stop.set()
        sampler.join()
        cp.cuda.set_allocator(allocator)
        peaks["used"] = max(peaks["used"], pool.used_bytes())
        peaks["reserved"] = max(peaks["reserved"], pool.total_bytes())
        stats = jb.runtime_stats()
        final_jax = device.memory_stats()
        warp_final = (
            warp.get_mempool_used_mem_current(warp_device) if warp_device else None
        )
        if warp_device:
            warp_peak[0] = max(warp_peak[0], warp_final)
        torch_peak_active = torch.cuda.max_memory_allocated() if torch else None
        torch_peak_reserved = torch.cuda.max_memory_reserved() if torch else None
        # All output values are checked against independently computed oracles.
        for i, value in enumerate(outputs):
            host_args = workload.argument_sets()[i % 2]
            reference_args = (
                sets[i % 2] if workload.name == "torch_convolution" else host_args
            )
            if allocation_control:
                oracle = host_args[-1] + np.array(1, dtype=host_args[-1].dtype)
            else:
                for _ in range(args.iterations):
                    oracle = workload.reference(*reference_args)
                    reference_args = (*reference_args[:-1], oracle)
            actual = (
                cp.asnumpy(
                    value if isinstance(value, cp.ndarray) else cp.from_dlpack(value)
                )
                if direct
                else np.asarray(value)
            )
            if workload.comparison_indices is not None:
                actual = actual[workload.comparison_indices]
            np.testing.assert_allclose(
                actual, oracle, rtol=workload.rtol, atol=workload.atol
            )
        analysis = compiled.memory_analysis() if compiled is not None else None
        result = {
            "package_version": jb.__version__,
            "package_path": str(Path(jb.__file__).resolve().parent),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "jax",
                    "jaxlib",
                    "numpy",
                    "cupy-cuda12x",
                    *(["warp-lang"] if warp_device else []),
                )
            },
            "application": workload.name,
            "size": args.size,
            "variant": args.worker_variant,
            "measurement_kind": "JAX output allocation control"
            if allocation_control
            else "External application",
            "gpu": gpu,
            "count": args.count,
            "iterations": args.iterations,
            "unroll": args.unroll,
            "workspace_bytes": args.workspace_bytes if not direct else 0,
            "trace_cache_size": args.trace_cache_size if graph else None,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "cupy_initial_active_bytes": initial_used,
            "cupy_peak_active_bytes": peaks["used"],
            "cupy_peak_additional_active_bytes": peaks["used"] - initial_used,
            "cupy_initial_reserved_bytes": initial_reserved,
            "cupy_peak_reserved_bytes": peaks["reserved"],
            "jax_initial_memory_stats": jax_initial,
            "jax_final_memory_stats": final_jax,
            "jax_executable_bytes": {
                key: getattr(analysis, key)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            }
            if analysis is not None
            else None,
            "driver_initial_used_bytes": driver_initial,
            "driver_sampled_peak_used_bytes": driver_peak[0],
            "runtime": stats,
            "torch_initial_active_bytes": torch_initial,
            "torch_peak_active_bytes": torch_peak_active,
            "torch_peak_reserved_bytes": torch_peak_reserved,
            "output_bytes_per_call": (
                cp.from_dlpack(outputs[0]).nbytes
                if direct and not isinstance(outputs[0], cp.ndarray)
                else outputs[0].nbytes
            ),
            "warp_initial_active_bytes": warp_initial,
            "warp_final_active_bytes": warp_final,
            "warp_sampled_peak_active_bytes": warp_peak[0],
            "warp_initial_all_time_high_bytes": warp_initial_high,
            "warp_final_all_time_high_bytes": warp.get_mempool_used_mem_high(
                warp_device
            )
            if warp_device
            else None,
            "native_graph_reserved_bytes": native_graph_storage.total_bytes()
            if native_graph_storage is not None
            else None,
            "native_graph_output_protocol": "Stable private graph outputs copied into a fresh result per invocation"
            if native_graph_storage is not None
            else None,
        }
        del outputs
        context.close()
        jb.synchronize(clear_cache=True)
        cp.cuda.Device().synchronize()
        return result
    finally:
        cp.cuda.set_allocator(allocator)
        stop.set()
        sampler.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apps",
        nargs="+",
        default=[
            "spectral_poisson",
            "image_blur",
            "row_reduction",
            "prefix_scan",
            "special_erf",
            "sort",
            "batched_gemm",
        ],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "direct_library",
            "direct_library_graph",
            "native_return",
            "jax_allocation_control",
            "jaxbind_gpu",
            "jaxbind_gpu_graph",
            "jaxbind_gpu_cached",
        ),
        default=["direct_library", "jaxbind_gpu", "jaxbind_gpu_graph"],
    )
    parser.add_argument("--size", choices=("small", "medium"), default="medium")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--workspace-bytes", type=int, default=0)
    parser.add_argument("--unroll", action="store_true")
    parser.add_argument("--trace-cache-size", type=int, default=4)
    parser.add_argument("--torch-path", type=Path)
    parser.add_argument("--worker-app")
    parser.add_argument(
        "--worker-variant",
        choices=(
            "direct_library",
            "direct_library_graph",
            "native_return",
            "jax_allocation_control",
            "jaxbind_gpu",
            "jaxbind_gpu_graph",
            "jaxbind_gpu_cached",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/memory-benchmarks.json",
    )
    args = parser.parse_args()
    if (
        args.count < 1
        or args.trace_cache_size < 1
        or args.iterations < 1
        or args.workspace_bytes < 0
    ):
        parser.error(
            "count, iterations and trace cache size must be positive; workspace bytes must be nonnegative"
        )
    if args.worker_app:
        print(json.dumps(worker(args)), flush=True)
        return
    results, skipped = [], []
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "methodology": __doc__,
        "results": results,
        "skipped": skipped,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "scripts/benchmark_common.py",
                ROOT / "src/jaxbind_gpu/_runtime.py",
                ROOT / "benchmarks/applications/operators.py",
                ROOT / "benchmarks/applications/workloads.py",
                ROOT / "benchmarks/applications/native_baselines.py",
                ROOT / "benchmarks/applications/warp_sparse.py",
                ROOT / "examples/warp_sparse.py",
                *sorted((ROOT / "src/jaxbind_gpu").glob("*.py")),
                *sorted((ROOT / "src/jaxbind_gpu").glob("*.so")),
                *sorted((ROOT / "cpp").glob("*.*")),
            )
        },
    }
    for app in args.apps:
        for variant in args.variants:
            if (
                variant in ("jaxbind_gpu_graph", "jaxbind_gpu_cached")
                and app == "batched_gemm"
            ):
                skipped.append(
                    {
                        "application": app,
                        "variant": variant,
                        "reason": "CuPy rejects its cuBLAS API during capture",
                    }
                )
                continue
            print(f"Memory {app}/{variant}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-app",
                app,
                "--worker-variant",
                variant,
                "--size",
                args.size,
                "--count",
                str(args.count),
                "--iterations",
                str(args.iterations),
                "--workspace-bytes",
                str(args.workspace_bytes),
                "--trace-cache-size",
                str(args.trace_cache_size),
            ]
            if args.torch_path:
                command.extend(["--torch-path", str(args.torch_path.resolve())])
            if args.unroll:
                command.append("--unroll")
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            results.append(json.loads(result.stdout.strip().splitlines()[-1]))
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
