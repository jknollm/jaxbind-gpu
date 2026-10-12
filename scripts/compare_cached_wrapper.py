"""Compare equal plan/geometry reuse through the wrapper's original JAX API."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_common import measure_pair

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/fair-wrapper-benchmarks.json"),
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.devices()
    sys.path.insert(0, str(args.reference_path.resolve()))
    import jax_finufft as reference
    from jax_finufft import jax_finufft_gpu
    from jax_finufft.options import Opts
    from examples import cufinufft_demo as ours
    import jaxbind_gpu as jb

    assert Path(jax_finufft_gpu.__file__).is_relative_to(args.reference_path.resolve())

    native_library = ctypes.CDLL(jax_finufft_gpu.__file__)
    mode = native_library.jax_finufft_benchmark_mode
    mode.argtypes = [ctypes.c_int]
    mode.restype = None
    rng = np.random.default_rng(512)
    results = []
    configurations = [
        ((4096,), 4096, 1),
        ((131072,), 131072, 1),
        ((131072,), 131072, 4),
        ((64, 64), 4096, 1),
        ((256, 256), 65536, 1),
        ((32, 32, 32), 32768, 1),
    ]
    payload = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_model=jax.devices("gpu")[0].device_kind,
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
        execution_package={
            "version": jb.__version__,
            "path": str(Path(jb.__file__).resolve().parent),
            "sha256": {
                str(p.relative_to(Path(jb.__file__).resolve().parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (
                    *Path(jb.__file__).resolve().parent.glob("*.py"),
                    *Path(jb.__file__).resolve().parent.glob("*.so"),
                )
            },
        },
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "scripts/compare_wrapper.py",
                ROOT / "scripts/benchmark_common.py",
                ROOT / "scripts/benchmark_applications.py",
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.py")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.cc")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.h")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.so")),
            )
        },
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "jaxbind-gpu",
                "jax",
                "jaxlib",
                "cufinufft",
                "numpy",
                "nanobind",
            )
        },
        reference_version=reference.__version__,
        reference_extension_sha256=hashlib.sha256(
            Path(jax_finufft_gpu.__file__).read_bytes()
        ).hexdigest(),
        reference_build="Benchmark variant of jax-finufft 1.3.1, with "
        "plan reuse and owned prepared geometry added to its native "
        "execution code. Python API and transformation rules unchanged.",
        methodology="Both implementations reuse plans and enqueue asynchronously "
        "on XLA streams. Prepared variants own coordinate copies and "
        "skip setpts after initialization. Dynamic variants alternate "
        "two GPU-resident coordinate/data sets. Prepared variants "
        "alternate data with immutable coordinates. Seven warmed "
        "batches of thirty calls, each immediately preceded by an untimed batch with the same protocol. Latency blocks every call, throughput "
        "blocks all outputs per batch. Variant order alternates within each "
        "geometry-reuse mode. Timed execution forbids device-to-host transfers. "
        "Cold call excludes compilation "
        "and includes plan construction. Every argument set is "
        "checked before timing. All settings and cuFINUFFT library match.",
        results=results,
    )
    for dtype in (np.complex64, np.complex128):
        real = np.float32 if dtype == np.complex64 else np.float64
        eps = 1e-5 if dtype == np.complex64 else 1e-10
        for shape, point_count, batch in configurations:
            points = tuple(
                jnp.asarray(rng.uniform(-np.pi, np.pi, point_count).astype(real))
                for _ in shape
            )
            other_points = tuple(p + real(0.013) for p in points)
            for kind in (1, 2):
                mode(0)
                data_shape = (point_count,) if kind == 1 else shape
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
                sign = 1 if kind == 1 else -1
                method = 1 if dtype == np.complex128 and len(shape) == 3 else 0
                opts = Opts(gpu_upsampfac=2.0, gpu_method=method)

                def external(data, *coordinates):
                    coordinates = tuple(
                        p[None, :] if batch > 1 else p for p in coordinates
                    )
                    if kind == 1:
                        return reference.nufft1(
                            shape, data, *coordinates, iflag=sign, eps=eps, opts=opts
                        )
                    return reference.nufft2(
                        data, *coordinates, iflag=sign, eps=eps, opts=opts
                    )

                def dynamic(data, *coordinates):
                    if kind == 1:
                        return ours.nufft1(
                            coordinates,
                            data,
                            shape,
                            isign=sign,
                            eps=eps,
                            upsampfac=2.0,
                            gpu_method=method,
                        )
                    return ours.nufft2(
                        coordinates,
                        data,
                        isign=sign,
                        eps=eps,
                        upsampfac=2.0,
                        gpu_method=method,
                    )

                prepare = ours.prepare_nufft1 if kind == 1 else ours.prepare_nufft2
                prepared = jax.jit(
                    prepare(
                        points,
                        *((shape,) if kind == 1 else ()),
                        isign=sign,
                        eps=eps,
                        upsampfac=2.0,
                        gpu_method=method,
                    )
                )
                compiled_ref = jax.jit(external)
                expected = np.asarray(compiled_ref(data, *points))
                expected_other = np.asarray(compiled_ref(other_data, *other_points))
                tolerance = 5e-5 if dtype == np.complex64 else 2e-9
                row = dict(
                    kind=kind,
                    grid=shape,
                    points=point_count,
                    batch=batch,
                    dtype=np.dtype(dtype).name,
                    eps=eps,
                    upsampfac=2.0,
                    gpu_method=method,
                    timings={},
                )
                variants = [
                    (
                        "wrapper_cached_dynamic",
                        compiled_ref,
                        [(data, *points), (other_data, *other_points)],
                        False,
                    ),
                    (
                        "jaxbind_gpu_dynamic",
                        jax.jit(dynamic),
                        [(data, *points), (other_data, *other_points)],
                        False,
                    ),
                    (
                        "wrapper_cached_prepared",
                        compiled_ref,
                        [(data, *points), (other_data, *points)],
                        True,
                    ),
                    ("jaxbind_gpu_prepared", prepared, [(data,), (other_data,)], True),
                ]
                for name, function, argument_sets, frozen in variants:
                    mode(int(frozen))
                    for index, arguments in enumerate(argument_sets):
                        target = (
                            expected
                            if index == 0
                            else expected * real(1.1)
                            if frozen
                            else expected_other
                        )
                        np.testing.assert_allclose(
                            function(*arguments), target, atol=tolerance, rtol=tolerance
                        )
                    # Compilation and correctness checks have completed. Clear
                    # resources so the first timed call measures cold plan setup.
                    ours.synchronize(clear_cache=True)
                    mode(int(frozen))
                for frozen in (False, True):
                    ours.synchronize(clear_cache=True)
                    mode(int(frozen))
                    paired = [variant for variant in variants if variant[3] == frozen]
                    row["timings"].update(measure_pair(paired))
                for suffix in ("dynamic", "prepared"):
                    baseline = row["timings"]["wrapper_cached_" + suffix]
                    actual = row["timings"]["jaxbind_gpu_" + suffix]
                    actual["latency_ratio"] = (
                        actual["latency_us"] / baseline["latency_us"]
                    )
                    actual["throughput_ratio"] = (
                        actual["throughput_us"] / baseline["throughput_us"]
                    )
                results.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                ours.synchronize(clear_cache=True)
                mode(0)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
