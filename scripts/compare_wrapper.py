"""Compare GPU-resident workloads against the released jax-finufft wrapper."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from benchmark_common import check_exclusive, measure_pair

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/wrapper-benchmarks.json"),
    )
    args = parser.parse_args()
    import cupy as cp

    jax.config.update("jax_enable_x64", True)
    jax.devices()
    sys.path.insert(0, str(args.reference_path.resolve()))
    import jax_finufft as reference
    from jax_finufft import jax_finufft_gpu
    from jax_finufft.options import Opts
    from examples import cufinufft_demo as ours
    import jaxbind_gpu as jb

    assert Path(jax_finufft_gpu.__file__).is_relative_to(args.reference_path.resolve())
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
        device=str(jax.devices("gpu")[0]),
        gpu_model=jax.devices("gpu")[0].device_kind,
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
        execution_package=dict(
            version=jb.__version__,
            path=str(Path(jb.__file__).resolve().parent),
            sha256={
                str(p.relative_to(Path(jb.__file__).resolve().parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (
                    *Path(jb.__file__).resolve().parent.glob("*.py"),
                    *Path(jb.__file__).resolve().parent.glob("*.so"),
                )
            },
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "scripts/benchmark_common.py",
                ROOT / "scripts/benchmark_applications.py",
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.py")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.cc")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.h")),
                *sorted((ROOT / "examples/cufinufft_demo").glob("*.so")),
            )
        },
        reference_version=reference.__version__,
        reference_extension_sha256=hashlib.sha256(
            Path(jax_finufft_gpu.__file__).read_bytes()
        ).hexdigest(),
        reference_build="Unmodified jax-finufft 1.3.1 GPU wrapper sources, compiled "
        "with c++ -O3 and linked against the installed cuFINUFFT 2.5.1.",
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
        methodology="Seven warmed batches of thirty calls, with an immediate untimed warm batch before each metric. Variant order alternates. Latency blocks after each "
        "call; throughput blocks all outputs after each batch. Two changing data/coordinate sets reside on the same GPU. Prepared geometry is fixed while data changes. Device-to-host transfers are disallowed during measurement. Compilation and transfers "
        "are excluded. Prepared geometry setup is excluded. Every "
        "variant is checked against the reference before timing.",
        results=results,
    )
    for dtype in (np.complex64, np.complex128):
        real = np.float32 if dtype == np.complex64 else np.float64
        eps = 1e-5 if dtype == np.complex64 else 1e-10
        for shape, point_count, batch in configurations:
            check_exclusive(cp)
            coordinates = tuple(
                jnp.asarray(rng.uniform(-np.pi, np.pi, point_count).astype(real))
                for _ in shape
            )
            for kind in (1, 2):
                data_shape = (point_count,) if kind == 1 else shape
                if batch > 1:
                    data_shape = (batch,) + data_shape
                data = jnp.asarray(
                    (
                        (rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape))
                        / np.sqrt(np.prod(data_shape))
                    ).astype(dtype)
                )
                sign = 1 if kind == 1 else -1
                gpu_method = 1 if dtype == np.complex128 and len(shape) == 3 else 0
                opts = Opts(gpu_upsampfac=2.0, gpu_method=gpu_method)

                def external(data, *points):
                    points = tuple(p[None, :] if batch > 1 else p for p in points)
                    if kind == 1:
                        return reference.nufft1(
                            shape, data, *points, iflag=sign, eps=eps, opts=opts
                        )
                    return reference.nufft2(
                        data, *points, iflag=sign, eps=eps, opts=opts
                    )

                def dynamic(data, *points):
                    if kind == 1:
                        return ours.nufft1(
                            points,
                            data,
                            shape,
                            isign=sign,
                            eps=eps,
                            upsampfac=2.0,
                            gpu_method=gpu_method,
                        )
                    return ours.nufft2(
                        points,
                        data,
                        isign=sign,
                        eps=eps,
                        upsampfac=2.0,
                        gpu_method=gpu_method,
                    )

                prepare = ours.prepare_nufft1 if kind == 1 else ours.prepare_nufft2
                prepared = prepare(
                    coordinates,
                    *((shape,) if kind == 1 else ()),
                    isign=sign,
                    eps=eps,
                    upsampfac=2.0,
                    gpu_method=gpu_method,
                )
                data_b = jnp.asarray(
                    (
                        (rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape))
                        / np.sqrt(np.prod(data_shape))
                    ).astype(dtype)
                )
                coordinates_b = tuple(
                    jnp.asarray(rng.uniform(-np.pi, np.pi, point_count).astype(real))
                    for _ in shape
                )
                dynamic_sets = [(data, *coordinates), (data_b, *coordinates_b)]
                variants = [
                    ("jax_finufft", jax.jit(external), dynamic_sets),
                    ("jaxbind_gpu_dynamic", jax.jit(dynamic), dynamic_sets),
                    ("jaxbind_gpu_prepared", jax.jit(prepared), [(data,), (data_b,)]),
                ]
                jax.block_until_ready(dynamic_sets)
                tolerance = 5e-5 if dtype == np.complex64 else 2e-9
                row = dict(
                    kind=kind,
                    grid=shape,
                    points=point_count,
                    batch=batch,
                    dtype=np.dtype(dtype).name,
                    eps=eps,
                    upsampfac=2.0,
                    gpu_method=gpu_method,
                    timings={},
                )
                for name, function, argument_sets in variants:
                    check_exclusive(cp)
                    for arguments in argument_sets:
                        points = (
                            coordinates if name.endswith("prepared") else arguments[1:]
                        )
                        expected = np.asarray(variants[0][1](arguments[0], *points))
                        actual = np.asarray(function(*arguments))
                        np.testing.assert_allclose(
                            actual, expected, atol=tolerance, rtol=tolerance
                        )
                row["timings"] = measure_pair(variants)
                baseline = row["timings"]["jax_finufft"]
                for name in ("jaxbind_gpu_dynamic", "jaxbind_gpu_prepared"):
                    value = row["timings"][name]
                    value["latency_speedup"] = (
                        baseline["latency_us"] / value["latency_us"]
                    )
                    value["throughput_speedup"] = (
                        baseline["throughput_us"] / value["throughput_us"]
                    )
                results.append(row)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                ours.synchronize(clear_cache=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
