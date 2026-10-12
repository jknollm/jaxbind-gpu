"""Compare direct and compiled public bindings with native CuPy/Torch execution."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

from benchmark_common import DirectContext, check_exclusive, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-path", type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 1048576])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/current-interfaces.json"),
    )
    args = parser.parse_args()
    if min(args.count, args.repeats, *args.sizes) < 1:
        parser.error("sizes, count and repeats must be positive")
    if args.torch_path:
        sys.path.insert(0, str(args.torch_path.resolve()))
    import cupy as cp
    import jax
    import numpy as np
    import torch
    import jaxbind_gpu as jb

    gpu = check_exclusive(cp)
    package = Path(jb.__file__).parent
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": jb.__version__,
        "gpu": gpu,
        "source_sha256": {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (*sorted(package.glob("*.py")), *sorted(package.glob("*.so")))
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_support_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_common.py").read_bytes()
        ).hexdigest(),
        "packages": {
            "jax": jax.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
        },
        "count": args.count,
        "repeats": args.repeats,
        "methodology": "The same library square operation, float32, changing resident inputs and fresh held outputs. Native Torch inputs and stream are prepared once. Every batch has an immediate warm batch, variant order alternates, and all work completes before timing ends. Compilation and independent NumPy checks occur outside timing. Direct calls include the binding's automatic JIT cache. Copies, direct output writes and recorded execution are separate variants.",
        "applications": [],
    }
    for backend, library in (("cupy", cp), ("torch", torch)):
        for size in args.sizes:
            check_exclusive(cp)
            sets = [
                (jax.device_put(np.linspace(0, 1, size, dtype=np.float32) + i),)
                for i in (0, 1)
            ]
            jax.block_until_ready(sets)
            context = DirectContext(cp)
            native_sets = [tuple(cp.from_dlpack(x) for x in s) for s in sets]
            native_stream = (
                context.stream
                if backend == "cupy"
                else torch.cuda.ExternalStream(context.stream.ptr, device=0)
            )
            if backend == "torch":
                native_sets = [
                    tuple(torch.utils.dlpack.from_dlpack(x) for x in s)
                    for s in native_sets
                ]

            def native(x):
                if backend == "cupy":
                    with native_stream:
                        return library.square(x)
                with torch.cuda.stream(native_stream), torch.no_grad():
                    return library.square(x)

            copy = jb.bind(
                library.square, backend=backend, jvp=lambda p, t: 2 * p[0] * t[0]
            )
            direct = jb.bind(
                library.square,
                backend=backend,
                out=True,
                jvp=lambda p, t: 2 * p[0] * t[0],
            )
            cached = jb.bind(
                library.square, backend=backend, out=True, graph_compatible=True
            )
            operations = {
                "eager_copy": copy,
                "jit_copy": jax.jit(copy),
                "eager_out": direct,
                "jit_out": jax.jit(direct),
                "cached_out": jax.jit(cached),
            }
            if backend == "cupy":
                operations["jit_allocate"] = jax.jit(
                    jb.bind(library.square, out="allocate")
                )
            variants = {
                "native": (native, native_sets, lambda outputs: context.synchronize())
            }
            variants.update(
                {
                    name: (op, sets, jax.block_until_ready)
                    for name, op in operations.items()
                }
            )
            for name, (fn, inputs, synchronize) in variants.items():
                for input_set, host_offset in zip(inputs, (0, 1)):
                    value = fn(*input_set)
                    synchronize([value])
                    actual = (
                        cp.asnumpy(value)
                        if isinstance(value, cp.ndarray)
                        else value.detach().cpu().numpy()
                        if isinstance(value, torch.Tensor)
                        else np.asarray(value)
                    )
                    expected = (
                        np.linspace(0, 1, size, dtype=np.float32) + host_offset
                    ) ** 2
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                for _ in range(6):
                    sample(fn, inputs, synchronize, count=args.count, latency=False)
            gradient = jax.grad(lambda x: direct(x).sum())(sets[0][0])
            np.testing.assert_allclose(
                np.asarray(gradient),
                2 * np.linspace(0, 1, size, dtype=np.float32),
                rtol=1e-6,
                atol=1e-7,
            )
            values = {name: {"latency": [], "throughput": []} for name in variants}
            with jax.transfer_guard_device_to_host("disallow"):
                for repeat in range(args.repeats):
                    check_exclusive(cp)
                    names = list(variants)
                    if repeat % 2:
                        names.reverse()
                    for name in names:
                        fn, inputs, synchronize = variants[name]
                        for metric in ("latency", "throughput"):
                            values[name][metric].append(
                                sample(
                                    fn,
                                    inputs,
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
            payload["applications"].append(
                {"application": backend + "_square", "size": size, "results": results}
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                backend,
                size,
                {name: round(t["throughput_us"], 2) for name, t in results.items()},
                flush=True,
            )
            context.close()
            jb.synchronize(clear_cache=True)


if __name__ == "__main__":
    main()
