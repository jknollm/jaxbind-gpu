"""Measure additive dispatch overhead with an identical external CUDA kernel.

Array length varies while dtype, rank, buffer count, and the operation remain
fixed. Copies and CUDA graph recapture are measured as separate execution paths.
Use run_gpu.py to select an exclusive device. No GPU is needed for --help.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

from benchmark_common import DirectContext, check_exclusive, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536, 262144, 1048576, 4194304, 16777216],
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument(
        "--eager", action="store_true", help="Also measure direct binding calls"
    )
    parser.add_argument("--allocation-forwarding", action="store_true")
    parser.add_argument(
        "--ffi-reference",
        action="store_true",
        help="Build a bare typed FFI baseline with a C++17 compiler",
    )
    parser.add_argument(
        "--derivatives",
        action="store_true",
        help="Compare JVP/VJP execution using the same kernel; requires --ffi-reference",
    )
    parser.add_argument("--max-python-overhead-percent", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/current-dispatch.json"),
    )
    args = parser.parse_args()
    if min(*args.sizes, args.count, args.repeats) < 1:
        parser.error("Sizes, count, and repeats must be positive")
    if args.derivatives and not args.ffi_reference:
        parser.error("--derivatives requires --ffi-reference")

    import cupy as cp
    import jax
    import numpy as np
    import jaxbind_gpu as jb

    device = jax.devices("gpu")[0]
    gpu = check_exclusive(cp)
    kernel = cp.RawKernel(
        r"""
extern "C" __global__ void scale(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2 * x[i];
}
""",
        "scale",
    )
    kernel.compile()

    def external(x, *, out):
        kernel(((x.size + 255) // 256,), (256,), (x, out, np.int32(x.size)))

    def returning(x):
        result = cp.empty_like(x)
        external(x, out=result)
        return result

    operations = {
        "native_binding": jb.bind(
            kernel,
            launch=(lambda x: ((x.size + 255) // 256,), (256,)),
            scalars=lambda x: (np.int32(x.size),),
        ),
        "python_out": jb.bind(external, out=True),
        "python_copy": jb.bind(returning),
    }
    if args.allocation_forwarding:
        operations["python_allocate"] = jb.bind(returning, out="allocate")
    if args.graphs:
        operations["native_binding_graph"] = operations["native_binding"]
        operations["python_cached"] = jb.bind(
            external,
            out=True,
            graph_compatible=True,
        )
        operations["python_graph"] = jb.bind(
            external,
            out=True,
            graph_compatible=True,
        )
    if args.eager:
        operations["native_eager"] = operations["native_binding"]
        operations["python_eager"] = operations["python_out"]
        if args.graphs:
            operations["python_cached_eager"] = operations["python_cached"]
    reference_directory = None
    reference_source = (
        Path(__file__).resolve().parents[1] / "benchmarks/dispatch_reference.cc"
    )
    if args.ffi_reference:
        import jaxlib

        reference_directory = tempfile.TemporaryDirectory(prefix="jaxbind-dispatch-")
        target = Path(reference_directory.name) / "reference.so"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-I" + str(Path(jaxlib.__file__).parent / "include"),
                str(reference_source),
                "-ldl",
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        reference_library = ctypes.CDLL(str(target))
        reference = jb.NativeFunction(
            "jaxbind_dispatch_reference",
            jax.ffi.pycapsule(reference_library.dispatch_reference),
            attributes=lambda: {"kernel": np.int64(kernel.kernel.ptr)},
            owner=(reference_library, kernel),
        )
        typed = jb.bind(reference, transpose=reference)
        operations["typed_binding"] = typed
        cabi = jb.NativeFunction.from_cabi(
            reference_library.dispatch_cabi,
            encode=lambda: np.int64(kernel.kernel.ptr).tobytes(),
            owner=(reference_library, kernel),
            graph_compatible=True,
        )
        cabi_linear = jb.bind(cabi, transpose=cabi)
        operations["cabi_binding"] = cabi_linear

        def bare_ffi(x):
            return jax.ffi.ffi_call(
                reference.name,
                jax.ShapeDtypeStruct(x.shape, x.dtype),
            )(x, kernel=np.int64(kernel.kernel.ptr))

        operations["bare_ffi"] = bare_ffi
        if args.graphs:
            operations["bare_ffi_graph"] = bare_ffi
            operations["typed_binding_graph"] = typed
            operations["cabi_binding_graph"] = cabi_linear
        if args.derivatives:
            import jax.numpy as jnp

            native = jb.NativeFunction.from_cuda_kernel(
                kernel,
                grid=lambda x: ((x.size + 255) // 256,),
                block=(256,),
                scalars=lambda x: (np.int32(x.size),),
            )
            native_linear = jb.bind(native, transpose=native)
            bare_forward = jax.custom_jvp(bare_ffi)
            bare_forward.defjvp(lambda p, t: (bare_ffi(p[0]), bare_ffi(t[0])))
            bare_reverse = jax.custom_vjp(bare_ffi)
            bare_reverse.defvjp(
                lambda x: (bare_ffi(x), None),
                lambda residual, c: (bare_ffi(c),),
            )
            for name, forward, reverse in (
                ("native_binding", native_linear, native_linear),
                ("typed_binding", typed, typed),
                ("cabi_binding", cabi_linear, cabi_linear),
                ("bare_ffi", bare_forward, bare_reverse),
            ):
                operations[name + "_jvp"] = lambda t, op=forward: jax.jvp(
                    op,
                    (jnp.zeros_like(t),),
                    (t,),
                )[1]
                operations[name + "_vjp"] = lambda c, op=reverse: jax.vjp(
                    op,
                    jnp.zeros_like(c),
                )[1](c)[0]
    package = Path(jb.__file__).parent
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "package_version": jb.__version__,
        "package_path": str(package),
        "source_sha256": {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (*sorted(package.glob("*.py")), *sorted(package.glob("*.so")))
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_support_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_common.py").read_bytes()
        ).hexdigest(),
        "ffi_reference_source_sha256": hashlib.sha256(
            reference_source.read_bytes()
        ).hexdigest()
        if args.ffi_reference
        else None,
        "packages": {
            "jax": jax.__version__,
            "cupy": cp.__version__,
            "numpy": np.__version__,
        },
        "count": args.count,
        "repeats": args.repeats,
        "derivative_protocol": "JVP tangents and VJP cotangents are the changing resident inputs. Linear scale derivatives use exactly one instance of the same external kernel, checked in HLO. No dense Jacobian is formed."
        if args.derivatives
        else None,
        "methodology": "Identical scale kernel, float32, one input and output, two changing GPU-resident input sets. Every path allocates a fresh output, retained until the batch completes. Direct native library launches use a private stream. Python-copy includes its additional device copy. Variant order rotates and reverses. Six warm batches precede timing, and each timed batch immediately follows an untimed batch with the same retention/completion protocol, after exclusivity checks. Compilation, first execution, and capture counts are separate. Paired differences are end-to-end costs, including JAX dispatch and allocation; they are not a measurement of only the C ABI bridge.",
        "results": [],
    }
    context = DirectContext(cp)
    options = {
        "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
        "xla_gpu_graph_min_graph_size": 1,
    }
    for size in args.sizes:
        check_exclusive(cp)
        sets = [
            (jax.device_put(np.arange(size, dtype=np.float32) + offset, device),)
            for offset in (0, 1)
        ]
        jax.block_until_ready(sets)
        cupy_sets = [tuple(cp.from_dlpack(x) for x in s) for s in sets]
        assert all(
            c[0].data.ptr == j[0].unsafe_buffer_pointer()
            for c, j in zip(cupy_sets, sets)
        )

        def direct(x):
            with context.stream:
                return returning(x)

        variants = {
            "direct_library": (direct, cupy_sets, lambda outputs: context.synchronize())
        }
        cold = {}
        custom_call_counts = {}
        executable_memory = {}
        for name, operation in operations.items():
            start = time.perf_counter()
            lowered = jax.jit(
                operation,
                compiler_options=options if name.endswith("_graph") else None,
            ).lower(sets[0][0])
            compiled = lowered.compile()
            execution = operation if name.endswith("_eager") else compiled
            analysis = compiled.memory_analysis()
            executable_memory[name] = {
                key: getattr(analysis, key)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            }
            custom_call_counts[name] = sum(
                "custom-call(" in line for line in compiled.as_text().splitlines()
            )
            if name.startswith(
                ("bare_ffi", "typed_binding", "native_binding", "cabi_binding")
            ):
                assert custom_call_counts[name] == 1, (name, lowered.as_text())
            compile_ms = (time.perf_counter() - start) * 1e3
            start = time.perf_counter()
            y = execution(*sets[0]).block_until_ready()
            cold[name] = {
                "compile_ms": compile_ms,
                "first_execution_us": (time.perf_counter() - start) * 1e6,
                "includes_automatic_compilation": name.endswith("_eager"),
            }
            del y
            variants[name] = (execution, sets, jax.block_until_ready)
        for function, input_sets, synchronize in variants.values():
            for inputs in input_sets:
                output = function(*inputs)
                synchronize([output])
                actual = (
                    cp.asnumpy(output)
                    if isinstance(output, cp.ndarray)
                    else np.asarray(output)
                )
                expected = np.arange(size, dtype=np.float32) + (
                    0 if inputs is input_sets[0] else 1
                )
                np.testing.assert_array_equal(actual, 2 * expected)
                del output
            for _ in range(6):
                sample(
                    function, input_sets, synchronize, count=args.count, latency=False
                )
        values = {name: {"latency": [], "throughput": []} for name in variants}
        captures = jb.runtime_stats()["graph_captures"]
        with jax.transfer_guard_device_to_host("disallow"):
            for repetition in range(args.repeats):
                check_exclusive(cp)
                names = list(variants)
                names = (
                    names[repetition % len(names) :] + names[: repetition % len(names)]
                )
                if repetition % 2:
                    names.reverse()
                for name in names:
                    function, input_sets, synchronize = variants[name]
                    for metric in ("latency", "throughput"):
                        values[name][metric].append(
                            sample(
                                function,
                                input_sets,
                                synchronize,
                                count=args.count,
                                latency=metric == "latency",
                            )
                        )
                check_exclusive(cp)
        results = {
            name: {
                metric + "_us": statistics.median(samples)
                for metric, samples in measurements.items()
            }
            | {"samples_us": measurements}
            for name, measurements in values.items()
        }
        overhead = {
            name: {
                metric: statistics.median(
                    [
                        x - y
                        for x, y in zip(
                            values[name][metric], values["direct_library"][metric]
                        )
                    ]
                )
                for metric in ("latency", "throughput")
            }
            for name in variants
            if name != "direct_library"
        }
        row = {
            "size": size,
            "array_bytes": size * 4,
            "cold": cold,
            "hlo_custom_calls": custom_call_counts,
            "jax_executable_bytes": executable_memory,
            "results": results,
            "paired_added_us": overhead,
            "paired_python_vs_native_us": {
                metric: statistics.median(
                    x - y
                    for x, y in zip(
                        values["python_out"][metric],
                        values["native_binding"][metric],
                    )
                )
                for metric in ("latency", "throughput")
            },
            "graph_captures_during_timing": jb.runtime_stats()["graph_captures"]
            - captures,
            "graph_capture_scope": "Measurement phase, including immediate untimed warm batches",
        }
        if args.ffi_reference:
            row["paired_native_vs_bare_ffi_us"] = {
                name: {
                    metric: statistics.median(
                        x - y
                        for x, y in zip(
                            values[name][metric], values["bare_ffi" + suffix][metric]
                        )
                    )
                    for metric in ("latency", "throughput")
                }
                for suffix in ("", "_jvp", "_vjp")
                if not suffix or args.derivatives
                for name in (
                    "native_binding" + suffix,
                    "typed_binding" + suffix,
                    "cabi_binding" + suffix,
                )
            }
        if args.graphs:
            row["paired_python_cached_vs_native_us"] = {
                metric: statistics.median(
                    x - y
                    for x, y in zip(
                        values["python_cached"][metric],
                        values["native_binding"][metric],
                    )
                )
                for metric in ("latency", "throughput")
            }
            row["paired_python_graph_vs_native_graph_us"] = {
                metric: statistics.median(
                    x - y
                    for x, y in zip(
                        values["python_graph"][metric],
                        values["native_binding_graph"][metric],
                    )
                )
                for metric in ("latency", "throughput")
            }
            if args.ffi_reference:
                row["paired_graph_binding_vs_bare_ffi_us"] = {
                    name: {
                        metric: statistics.median(
                            x - y
                            for x, y in zip(
                                values[name][metric], values["bare_ffi_graph"][metric]
                            )
                        )
                        for metric in ("latency", "throughput")
                    }
                    for name in (
                        "native_binding_graph",
                        "typed_binding_graph",
                        "cabi_binding_graph",
                        "python_graph",
                    )
                }
        payload["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"size": size, "paired_added_us": overhead}), flush=True)
        jb.synchronize(clear_cache=True)
    if len(args.sizes) > 1:
        x = np.array([row["array_bytes"] / (1 << 20) for row in payload["results"]])
        fits = {}
        for name in operations:
            fits[name] = {}
            for metric in ("latency", "throughput"):
                y = np.array(
                    [row["paired_added_us"][name][metric] for row in payload["results"]]
                )
                slope, intercept = np.polyfit(x, y, 1)
                fits[name][metric] = {
                    "constant_fit_us": float(np.mean(y)),
                    "constant_fit_rms_us": float(
                        np.sqrt(np.mean((y - np.mean(y)) ** 2))
                    ),
                    "linear_fit_us_per_mib": float(slope),
                    "linear_fit_intercept_us": float(intercept),
                    "linear_fit_rms_us": float(
                        np.sqrt(np.mean((y - (slope * x + intercept)) ** 2))
                    ),
                    "min_added_us": float(min(y)),
                    "max_added_us": float(max(y)),
                }
        payload["size_scaling"] = fits
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    context.close()
    if reference_directory is not None:
        reference_directory.cleanup()
    if args.max_python_overhead_percent is not None:
        ratios = [
            r["results"]["python_out"]["throughput_us"]
            / r["results"]["native_binding"]["throughput_us"]
            - 1
            for r in payload["results"]
        ]
        assert max(ratios) * 100 <= args.max_python_overhead_percent, (
            f"Python overhead vs identical native binding: {max(ratios) * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
