"""Check and benchmark real external GPU applications on an exclusive GPU.

Use scripts/run_gpu.py to select an idle device. All imports of optional
application libraries occur after argument parsing; --list needs no GPU.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from benchmark_common import DirectContext, check_exclusive, sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))


def run(
    workload,
    *,
    count,
    repeats,
    check_only,
    cp,
    jax,
    np,
    graphs=False,
    copy_results=False,
    iterations=1,
    jax_reference=False,
    native_graphs=False,
    unroll=False,
    workspace_bytes=0,
    native_return=False,
    cached=False,
):
    import jaxbind_gpu as jb

    device = jax.devices("gpu")[0]
    argument_sets = [
        tuple(jax.device_put(a, device) for a in args)
        for args in workload.argument_sets()
    ]
    jax.block_until_ready(argument_sets)
    if workspace_bytes:
        if not hasattr(workload.operator, "workspace_bytes"):
            raise ValueError("This operator manages its own native workspace")
        workload.operator.workspace_bytes = workspace_bytes
        workload.operator.enable_graph(graphs or cached)
    if copy_results:
        if not hasattr(workload.operator, "writes_output"):
            raise ValueError(f"{workload.name} uses its existing output interface")
        workload.operator.writes_output = False
        workload.operator.enable_graph(graphs or cached)
    if graphs or cached:
        if not hasattr(workload.operator, "enable_graph"):
            raise ValueError(
                f"{workload.name} does not supply a capture-safe implementation"
            )
        workload.operator.enable_graph()
    if iterations > 1:
        spec = jax.eval_shape(workload.bound, *argument_sets[0])
        if (spec.shape, spec.dtype) != (
            argument_sets[0][-1].shape,
            argument_sets[0][-1].dtype,
        ):
            raise ValueError(
                "Iteration benchmarks require the output to match the last input"
            )

    def bound_program(*args):
        if iterations == 1:
            return workload.bound(*args)
        return jax.lax.fori_loop(
            0,
            iterations,
            lambda i, x: workload.bound(*args[:-1], x),
            args[-1],
            unroll=unroll,
        )

    start = time.perf_counter()
    options = (
        {
            "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL,WHILE",
            "xla_gpu_graph_min_graph_size": 1,
        }
        if graphs
        else None
    )
    compiled = (
        jax.jit(bound_program, compiler_options=options)
        .lower(*argument_sets[0])
        .compile()
    )
    compile_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    with jax.transfer_guard_device_to_host("disallow"):
        value = compiled(*argument_sets[0]).block_until_ready()
    first_execution_us = (time.perf_counter() - start) * 1e6
    observed = getattr(workload.operator, "observed", None)
    if observed is not None and workload.steps == 1 and iterations == 1:
        assert observed["inputs"] == tuple(
            a.unsafe_buffer_pointer() for a in argument_sets[0]
        )
        assert observed["output"] == value.unsafe_buffer_pointer()
        assert observed["stream"] != 0

    context = DirectContext(cp)
    cupy_args = [tuple(cp.from_dlpack(a) for a in args) for args in argument_sets]
    assert all(
        c.data.ptr == j.unsafe_buffer_pointer()
        for ca, ja in zip(cupy_args, argument_sets)
        for c, j in zip(ca, ja)
    )

    def bound(*args):
        return compiled(*args)

    def direct(*args):
        with context.stream:
            for _ in range(iterations):
                value = workload.direct(args, context)
                args = (*args[:-1], value)
            return value

    def sync_bound(outputs=None):
        # This waits for both XLA dispatch and device execution. A device-only
        # synchronization could precede asynchronous dispatch and undercount it.
        if outputs is not None:
            jax.block_until_ready(outputs)
        else:
            jax.effects_barrier()
            cp.cuda.Device().synchronize()

    errors = []
    for host_args, args, ca in zip(workload.argument_sets(), argument_sets, cupy_args):
        with jax.transfer_guard_device_to_host("disallow"):
            actual = bound(*args).block_until_ready()
        direct_value = direct(*ca)
        context.synchronize()
        host_value = np.asarray(actual)
        cp_value = cp.asnumpy(direct_value)
        np.testing.assert_allclose(
            host_value, cp_value, rtol=workload.rtol, atol=workload.atol
        )
        reference_args = args if workload.name == "torch_convolution" else host_args
        for _ in range(iterations):
            oracle = workload.reference(*reference_args)
            reference_args = (*reference_args[:-1], oracle)
        oracle = np.asarray(oracle)
        checked = (
            host_value
            if workload.comparison_indices is None
            else host_value[workload.comparison_indices]
        )
        np.testing.assert_allclose(
            checked, oracle, rtol=workload.rtol, atol=workload.atol
        )
        errors.append(float(np.max(np.abs(checked - oracle))))

    variants = {
        "jaxbind_gpu": (bound, argument_sets, sync_bound),
        "direct_library": (direct, cupy_args, lambda outputs: context.synchronize()),
    }
    if native_return:
        if iterations != 1:
            raise ValueError(
                "Native return baselines currently measure one application"
            )
        from applications.native_baselines import returning_baseline

        baseline = returning_baseline(workload, cupy_args, context)
        if baseline is not None:
            function, sets = baseline
            for args, expected_args in zip(sets, cupy_args):
                value = function(*args)
                expected = direct(*expected_args)
                context.synchronize()
                actual = (
                    cp.from_dlpack(value)
                    if not isinstance(value, cp.ndarray)
                    else value
                )
                np.testing.assert_allclose(
                    cp.asnumpy(actual),
                    cp.asnumpy(expected),
                    rtol=workload.rtol,
                    atol=workload.atol,
                )
            variants["native_return"] = (
                function,
                sets,
                lambda outputs: context.synchronize(),
            )
    native_graph_storage = None
    if native_graphs:
        native_graph_storage = cp.cuda.MemoryPool()
        captures = []
        for args in cupy_args:
            with context.stream, cp.cuda.using_allocator(native_graph_storage.malloc):
                context.stream.begin_capture()
                value = direct(*args)
                graph = context.stream.end_capture()
            captures.append((graph, value))

        def direct_graph(*args):
            # Two variants may share fixed inputs. The last input always differs.
            index = 0 if args[-1].data.ptr == cupy_args[0][-1].data.ptr else 1
            graph, value = captures[index]
            with context.stream:
                graph.launch(stream=context.stream)
                return value.copy()

        for args in cupy_args:
            value = direct_graph(*args)
            expected = direct(*args)
            context.synchronize()
            np.testing.assert_allclose(
                cp.asnumpy(value),
                cp.asnumpy(expected),
                rtol=workload.rtol,
                atol=workload.atol,
            )
        variants["native_cuda_graph_with_output_copy"] = (
            direct_graph,
            cupy_args,
            lambda outputs: context.synchronize(),
        )
    if jax_reference:
        from applications.workloads import builtin_reference

        reference = builtin_reference(workload.name)

        def reference_program(*args):
            if iterations == 1:
                return reference(*args)
            return jax.lax.fori_loop(
                0,
                iterations,
                lambda i, x: reference(*args[:-1], x),
                args[-1],
                unroll=unroll,
            )

        builtin = jax.jit(reference_program).lower(*argument_sets[0]).compile()
        for args in argument_sets:
            np.testing.assert_allclose(
                np.asarray(builtin(*args)),
                np.asarray(bound(*args)),
                rtol=workload.rtol,
                atol=workload.atol,
            )
        variants["jax_builtin"] = (builtin, argument_sets, sync_bound)
    captures_before = jb.runtime_stats()["graph_captures"]
    results = {
        name: {"latency_samples_us": [], "throughput_samples_us": []}
        for name in variants
    }
    if not check_only:
        for function, sets, synchronize in variants.values():
            outputs = []
            for i in range(5):
                outputs.append(function(*sets[i % len(sets)]))
            synchronize(outputs)
        for repetition in range(repeats):
            # Alternate order to reduce drift bias. Each variant uses the exact
            # same warmed operation, precision, options, and reusable plans.
            names = list(variants)
            if repetition % 2:
                names.reverse()
            for name in names:
                check_exclusive(cp)
                function, sets, synchronize = variants[name]
                for metric, latency in (("latency", True), ("throughput", False)):
                    results[name][f"{metric}_samples_us"].append(
                        sample(
                            function, sets, synchronize, count=count, latency=latency
                        )
                    )
                check_exclusive(cp)
        for result in results.values():
            for metric in ("latency", "throughput"):
                result[f"{metric}_us"] = statistics.median(
                    result[f"{metric}_samples_us"]
                )
    sync_bound()
    context.close()
    jb.synchronize(clear_cache=True)
    return {
        "application": workload.name,
        "size": workload.size,
        "backend": workload.backend,
        "cuda_graphs": graphs,
        "callback_recording_cache": cached,
        "copy_results": copy_results,
        "steps": workload.steps,
        "iterations": iterations,
        "unroll": unroll,
        "workspace_bytes": workspace_bytes,
        "graph_captures_during_timing": jb.runtime_stats()["graph_captures"]
        - captures_before,
        "graph_capture_scope": "Initial variant warmup, immediate metric warmups, and timed batches",
        "direct_library_contract": "Same adapter algorithms, output allocation and result copies as the binding",
        "native_return_contract": "Fresh native result without adapter output allocation/copy; native tensors and stream prepared once for Torch and Warp"
        if "native_return" in variants
        else None,
        "native_graph_reserved_bytes": native_graph_storage.total_bytes()
        if native_graph_storage is not None
        else None,
        "native_graph_output_protocol": "Replay uses stable private outputs and copies to a fresh output on every call; this extra copy preserves held-output semantics"
        if native_graphs
        else None,
        "inputs": [
            {"shape": list(a.shape), "dtype": str(a.dtype)} for a in argument_sets[0]
        ],
        "compile_ms": compile_ms,
        "first_execution_us": first_execution_us,
        "oracle_max_absolute_errors": errors,
        "oracle_scope": "first 16 output points"
        if workload.comparison_indices is not None
        else "all outputs",
        "gpu_residency": "DLPack input pointers identical; JAX device-to-host transfer guard disallowed during bound execution",
        "pointer_audit": "callback input/output addresses match XLA"
        if observed is not None and workload.steps == 1 and iterations == 1
        else "core native pointer tests"
        if workload.backend == "Native C ABI"
        else "individual sparse adapter verified by acceptance tests",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--apps", nargs="+", help="Application names; default all available"
    )
    parser.add_argument(
        "--sizes", nargs="+", choices=("small", "medium"), default=["small", "medium"]
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--native-return",
        action="store_true",
        help="Also time fresh native results for sort, solve, sparse, Torch and Warp",
    )
    parser.add_argument("--workspace-bytes", type=int, default=0)
    parser.add_argument(
        "--unroll",
        action="store_true",
        help="Unroll the compiled loop for whole-program CUDA graph capture",
    )
    parser.add_argument(
        "--native-graphs",
        action="store_true",
        help="Also time direct CUDA graph replay with a fresh-output copy",
    )
    parser.add_argument(
        "--jax-reference",
        action="store_true",
        help="Also time an equivalent JAX builtin implementation where available",
    )
    parser.add_argument(
        "--copy-results",
        action="store_true",
        help="Measure the default return-value binding instead of direct out writes",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Opt in to XLA CUDA graph replay for capture-safe applications",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Reuse capture-safe callback recordings under ordinary jax.jit",
    )
    parser.add_argument(
        "--torch-path", type=Path, help="Optional isolated Torch installation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/application-benchmarks.json",
    )
    args = parser.parse_args()
    if (
        args.count < 1
        or args.repeats < 1
        or args.iterations < 1
        or args.workspace_bytes < 0
    ):
        parser.error(
            "count, repeats and iterations must be positive; workspace bytes must be nonnegative"
        )
    if args.torch_path:
        sys.path.insert(0, str(args.torch_path.resolve()))
    from applications.workloads import APPLICATIONS, build

    if args.list:
        for name, description in APPLICATIONS.items():
            print(f"{name}: {description}")
        return
    names = list(APPLICATIONS) if args.apps is None else args.apps
    unknown = set(names) - APPLICATIONS.keys()
    if unknown:
        parser.error(f"Unknown applications: {sorted(unknown)}")
    import jax
    import cupy as cp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    jax.devices("gpu")
    gpu = check_exclusive(cp)
    import jaxbind_gpu as jb

    package_path = Path(jb.__file__).resolve().parent
    skipped = []
    results = []
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "jaxbind-gpu",
                "jax",
                "jaxlib",
                "cupy-cuda12x",
                "numpy",
                "scipy",
            )
        },
        "execution_package": {
            "path": str(package_path),
            "version": jb.__version__,
            "sha256": {
                str(p.relative_to(package_path)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (*package_path.rglob("*.py"), *package_path.glob("*.so"))
            },
        },
        "count": args.count,
        "repeats": args.repeats,
        "check_only": args.check_only,
        "methodology": "Warmed end-to-end wall time including output allocation, host dispatch and completion; latency waits after every call, throughput waits after each batch. Each timed batch immediately follows an untimed batch with the same output-retention and completion protocol, after exclusivity checks. Two changing data sets are GPU resident. Variant order alternates. Bound and direct paths use the same external algorithms, precision and reusable plans. Direct out writes avoid return-value copies when supported; --copy-results measures that generic binding cost separately. Native library allocators reuse temporary storage in stream order. CUDA graphs are opt-in and measured separately. First bound execution and compilation are excluded from warmed measurements and reported separately. Native cuFINUFFT uses an equivalent reused direct plan with setpts on every call.",
        "skipped": skipped,
        "applications": results,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "benchmarks/applications/operators.py",
                ROOT / "benchmarks/applications/workloads.py",
                ROOT / "benchmarks/applications/native_baselines.py",
                ROOT / "benchmarks/applications/warp_sparse.py",
                ROOT / "examples/warp_sparse.py",
                Path(__file__).resolve(),
                ROOT / "scripts/benchmark_common.py",
                *sorted((ROOT / "src/jaxbind_gpu").glob("*.py")),
                *sorted((ROOT / "cpp").glob("*.*")),
                *sorted((ROOT / "src/jaxbind_gpu").glob("*.so")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.py")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.cc")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.h")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.so")),
            )
        },
    }
    for name in names:
        if name == "nufft_synthesis":
            payload["packages"]["cufinufft"] = importlib.metadata.version("cufinufft")
        if name == "warp_sparse":
            try:
                import warp
            except ImportError:
                if args.apps is not None:
                    raise
                skipped.append({"application": name, "reason": "Install warp-lang"})
                continue
            payload["packages"]["warp-lang"] = warp.__version__
        if name == "torch_convolution":
            try:
                import torch
            except ImportError:
                if args.apps is not None:
                    raise
                skipped.append(
                    {
                        "application": name,
                        "reason": "Install optional CUDA PyTorch or supply --torch-path",
                    }
                )
                continue
            payload["packages"]["torch"] = torch.__version__
            payload["torch_cudnn_version"] = torch.backends.cudnn.version()
            payload["torch_installation"] = (
                "isolated CUDA Torch installation"
                if args.torch_path
                else "installed optional dependency"
            )
        for size in args.sizes:
            check_exclusive(cp)
            print(f"Checking {name}/{size}", flush=True)
            result = run(
                build(name, size),
                count=args.count,
                repeats=args.repeats,
                check_only=args.check_only,
                cp=cp,
                jax=jax,
                np=np,
                graphs=args.graphs,
                copy_results=args.copy_results,
                iterations=args.iterations,
                jax_reference=args.jax_reference,
                native_graphs=args.native_graphs,
                unroll=args.unroll,
                workspace_bytes=args.workspace_bytes,
                native_return=args.native_return,
                cached=args.cached,
            )
            results.append(result)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            if not args.check_only:
                timings = result["results"]
                direct = timings["direct_library"]["throughput_us"]
                bound = timings["jaxbind_gpu"]["throughput_us"]
                print(
                    f"{name}/{size}: direct {direct:.1f} us, bound {bound:.1f} us, ratio {bound / direct:.3f}",
                    flush=True,
                )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
