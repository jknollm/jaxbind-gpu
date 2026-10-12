"""Isolate bridge overhead with the same cuFINUFFT native provider and plans."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from benchmark_common import check_exclusive, sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/current-provider.json")
    )
    args = parser.parse_args()
    if min(args.count, args.repeats) < 1:
        parser.error("count and repeats must be positive")
    import cupy as cp
    import jax
    import jaxbind_gpu as jb
    import numpy as np
    from examples import cufinufft_demo as demo
    from examples.cufinufft_demo import _bridge, _native

    jax.config.update("jax_enable_x64", True)
    gpu = check_exclusive(cp)
    with tempfile.TemporaryDirectory(prefix="jaxbind-provider-") as temporary:
        library_path = Path(temporary) / "reference.so"
        source = ROOT / "benchmarks/provider_reference.cc"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-I" + jax.ffi.include_dir(),
                str(source),
                "-o",
                str(library_path),
            ],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
        pointer = _bridge.cufinufft_address()
        reference = jb.NativeFunction(
            "jaxbind_provider_reference",
            jax.ffi.pycapsule(library.provider_reference),
            attributes=lambda **kw: {
                "address": np.int64(pointer),
                "config": _native._encode(**kw),
            },
            owner=(library, _bridge),
        )
        package = Path(jb.__file__).parent
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": jb.__version__,
            "gpu": gpu,
            "source_sha256": {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (*sorted(package.glob("*.py")), *sorted(package.glob("*.so")))
            },
            "benchmark_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    source,
                    Path(__file__),
                    ROOT / "scripts/benchmark_common.py",
                    *sorted((ROOT / "examples/cufinufft_demo").glob("*.py")),
                    *sorted((ROOT / "examples/cufinufft_demo").glob("*.cc")),
                    *sorted((ROOT / "examples/cufinufft_demo").glob("*.so")),
                )
            },
            "count": args.count,
            "repeats": args.repeats,
            "methodology": "Both variants call the identical provider address with identical configuration, coordinates, coefficients, output ownership and plan cache. Only the native FFI entry point changes. Minimal FFI still builds C ABI buffer descriptors. Compilation and correctness checks precede timing; changing resident inputs, fresh held outputs, immediate warm batches and alternating order. This isolates the generic bridge rather than comparing different adapters.",
            "results": [],
        }
        rng = np.random.default_rng(734)
        for shape in ((4096,), (131072,), (256, 256)):
            points = tuple(
                jax.device_put(
                    rng.uniform(-np.pi, np.pi, np.prod(shape)).astype(np.float64)
                )
                for _ in shape
            )
            inputs = [
                jax.device_put(
                    (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
                        np.complex128
                    )
                )
                for _ in range(2)
            ]
            jax.block_until_ready((points, inputs))
            for prepared in (False, True):
                if prepared:
                    operation = demo.prepare_nufft2(points, eps=1e-10, upsampfac=2.0)
                    sets = [(x,) for x in inputs]
                else:

                    def operation(x, *coordinates):
                        return demo.nufft2(coordinates, x, eps=1e-10, upsampfac=2.0)

                    sets = [
                        (inputs[i], *(p + i * 0.013 for p in points)) for i in range(2)
                    ]
                jax.block_until_ready(sets)
                _native._operation = None
                bound = jax.jit(lambda *xs: operation(*xs)).lower(*sets[0]).compile()
                original = _native._operation
                _native._operation = reference
                try:
                    bare = jax.jit(lambda *xs: operation(*xs)).lower(*sets[0]).compile()
                finally:
                    _native._operation = original
                assert "jaxbind_provider_reference" in bare.as_text()
                assert "jaxbind_provider_reference" not in bound.as_text()
                compiler_memory = {
                    name: {
                        field: getattr(executable.memory_analysis(), field)
                        for field in (
                            "argument_size_in_bytes",
                            "output_size_in_bytes",
                            "temp_size_in_bytes",
                        )
                    }
                    for name, executable in (
                        ("minimal_ffi", bare),
                        ("jaxbind_gpu", bound),
                    )
                }
                assert compiler_memory["minimal_ffi"] == compiler_memory["jaxbind_gpu"]
                variants = {"minimal_ffi": bare, "jaxbind_gpu": bound}
                for arguments in sets:
                    np.testing.assert_allclose(
                        bound(*arguments), bare(*arguments), rtol=1e-12, atol=1e-12
                    )
                measurements = {
                    name: {"latency": [], "throughput": []} for name in variants
                }
                with jax.transfer_guard_device_to_host("disallow"):
                    for repeat in range(args.repeats):
                        check_exclusive(cp)
                        names = list(variants)
                        if repeat % 2:
                            names.reverse()
                        for name in names:
                            for metric in ("latency", "throughput"):
                                measurements[name][metric].append(
                                    sample(
                                        variants[name],
                                        sets,
                                        jax.block_until_ready,
                                        count=args.count,
                                        latency=metric == "latency",
                                    )
                                )
                results = {
                    name: {
                        metric + "_us": statistics.median(values)
                        for metric, values in data.items()
                    }
                    | {"samples_us": data}
                    for name, data in measurements.items()
                }
                paired = {
                    metric: float(
                        np.median(
                            np.array(measurements["jaxbind_gpu"][metric])
                            - np.array(measurements["minimal_ffi"][metric])
                        )
                    )
                    for metric in ("latency", "throughput")
                }
                payload["results"].append(
                    {
                        "shape": shape,
                        "prepared": prepared,
                        "dtype": "complex128",
                        "kind": 2,
                        "results": results,
                        "paired_added_us": paired,
                        "compiler_memory": compiler_memory,
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(shape, prepared, paired, flush=True)
                demo.synchronize(clear_cache=True)


if __name__ == "__main__":
    main()
