"""Compare nonlinear forward, JVP and VJP calls with identical CUDA kernels.

The bound and bare FFI paths use the same sin/cos kernels and derivative recipe.
Direct-library calls use the same kernels and fresh output allocations. No
dense Jacobian is constructed. Inputs and directions change on the GPU.
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

from benchmark_common import DirectContext, check_exclusive, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 1048576])
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/current-nonlinear.json"),
    )
    args = parser.parse_args()
    if min(*args.sizes, args.count, args.repeats) < 1:
        parser.error("Sizes, count and repeats must be positive")

    import cupy as cp
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    import jaxbind_gpu as jb

    kernels = {}
    for name, signature, expression in (
        ("forward", "const float* x, float* y, int n", "sinf(x[i])"),
        (
            "derivative",
            "const float* x, const float* t, float* y, int n",
            "cosf(x[i]) * t[i]",
        ),
    ):
        kernel = cp.RawKernel(
            f'extern "C" __global__ void {name}({signature}) '
            "{ int i = blockIdx.x * blockDim.x + threadIdx.x; "
            f"if (i < n) y[i] = {expression}; }}",
            name,
        )
        kernel.compile()
        kernels[name] = kernel
    root = Path(__file__).resolve().parents[1]
    source = root / "benchmarks/nonlinear_reference.cc"
    package = Path(jb.__file__).resolve().parent
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": jb.__version__,
        "gpu": check_exclusive(cp),
        "count": args.count,
        "repeats": args.repeats,
        "methodology": __doc__,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), source)
        },
        "package_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (*package.glob("*.py"), *package.glob("*.so"))
        },
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="jaxbind-nonlinear-") as directory:
        target = Path(directory) / "reference.so"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-I" + str(Path(jaxlib.__file__).parent / "include"),
                str(source),
                "-ldl",
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        library = ctypes.CDLL(str(target))
        jax.ffi.register_ffi_target(
            "jaxbind_nonlinear_reference",
            jax.ffi.pycapsule(library.nonlinear_reference),
            platform="CUDA",
        )

        def bare(name, *inputs):
            return jax.ffi.ffi_call(
                "jaxbind_nonlinear_reference",
                jax.ShapeDtypeStruct(inputs[0].shape, inputs[0].dtype),
            )(*inputs, kernel=np.int64(kernels[name].kernel.ptr))

        def native(name):
            return jb.NativeFunction.from_cuda_kernel(
                kernels[name],
                grid=lambda x, *rest: ((x.size + 255) // 256,),
                block=(256,),
                scalars=lambda x, *rest: (np.int32(x.size),),
            )

        derivative = jb.bind(
            native("derivative"),
            jvp=lambda p, t: -jnp.sin(p[0]) * t[0] * p[1] + jnp.cos(p[0]) * t[1],
        )
        bound = jb.bind(
            native("forward"),
            jvp=lambda p, t: derivative(p[0], t[0]),
            vjp=lambda p, c: (derivative(p[0], c),),
        )
        bare_forward = jax.custom_jvp(lambda x: bare("forward", x))
        bare_forward.defjvp(
            lambda p, t: (bare("forward", p[0]), bare("derivative", p[0], t[0]))
        )
        bare_reverse = jax.custom_vjp(lambda x: bare("forward", x))
        bare_reverse.defvjp(
            lambda x: (bare("forward", x), x),
            lambda x, c: (bare("derivative", x, c),),
        )
        context = DirectContext(cp)
        for size in args.sizes:
            check_exclusive(cp)
            host = [
                (
                    np.linspace(-2, 2, size, dtype=np.float32) + offset,
                    np.full(size, 0.5 + offset, dtype=np.float32),
                )
                for offset in (0, 0.125)
            ]
            resident = [tuple(jax.device_put(a) for a in arrays) for arrays in host]
            jax.block_until_ready(resident)
            cupy = [tuple(cp.from_dlpack(a) for a in arrays) for arrays in resident]
            for mode in ("forward", "jvp", "vjp"):

                def transformed(op, x, direction):
                    if mode == "forward":
                        return op(x)
                    if mode == "jvp":
                        return jax.jvp(op, (x,), (direction,))[1]
                    return jax.vjp(op, x)[1](direction)[0]

                reference = bare_reverse if mode == "vjp" else bare_forward
                variants, memories, calls = {}, {}, {}
                for name, op in (("bound", bound), ("bare_ffi", reference)):
                    compiled = (
                        jax.jit(lambda x, t: transformed(op, x, t))
                        .lower(*resident[0])
                        .compile()
                    )
                    calls[name] = sum(
                        "custom-call(" in line
                        for line in compiled.as_text().splitlines()
                    )
                    assert calls[name] == 1
                    memory = compiled.memory_analysis()
                    memories[name] = {
                        key: getattr(memory, key)
                        for key in (
                            "argument_size_in_bytes",
                            "output_size_in_bytes",
                            "temp_size_in_bytes",
                            "alias_size_in_bytes",
                        )
                    }
                    variants[name] = (compiled, resident, jax.block_until_ready)

                def direct(x, direction):
                    with context.stream:
                        output = cp.empty_like(x)
                        name = "forward" if mode == "forward" else "derivative"
                        inputs = (x,) if mode == "forward" else (x, direction)
                        kernels[name](
                            ((size + 255) // 256,),
                            (256,),
                            (*inputs, output, np.int32(size)),
                        )
                        return output

                variants["direct_library"] = (
                    direct,
                    cupy,
                    lambda outputs: context.synchronize(),
                )
                for function, inputs, synchronize in variants.values():
                    for arrays, expected_inputs in zip(inputs, host):
                        output = function(*arrays)
                        synchronize([output])
                        actual = (
                            cp.asnumpy(output)
                            if isinstance(output, cp.ndarray)
                            else np.asarray(output)
                        )
                        x, t = expected_inputs
                        expected = np.sin(x) if mode == "forward" else np.cos(x) * t
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-6, atol=2e-6
                        )
                        del output
                    for _ in range(6):
                        sample(
                            function,
                            inputs,
                            synchronize,
                            count=args.count,
                            latency=False,
                        )
                samples = {
                    name: {metric: [] for metric in ("latency", "throughput")}
                    for name in variants
                }
                with jax.transfer_guard_device_to_host("disallow"):
                    for repetition in range(args.repeats):
                        check_exclusive(cp)
                        names = list(variants)
                        names = names[repetition % 3 :] + names[: repetition % 3]
                        if repetition % 2:
                            names.reverse()
                        for name in names:
                            function, inputs, synchronize = variants[name]
                            for metric in samples[name]:
                                samples[name][metric].append(
                                    sample(
                                        function,
                                        inputs,
                                        synchronize,
                                        count=args.count,
                                        latency=metric == "latency",
                                    )
                                )
                        check_exclusive(cp)
                row = {
                    "size": size,
                    "mode": mode,
                    "custom_calls": calls,
                    "jax_executable_bytes": memories,
                    "results": {
                        name: {
                            metric + "_us": statistics.median(values)
                            for metric, values in metrics.items()
                        }
                        | {"samples_us": metrics}
                        for name, metrics in samples.items()
                    },
                    "paired_bound_vs_bare_ffi_us": {
                        metric: statistics.median(
                            x - y
                            for x, y in zip(
                                samples["bound"][metric], samples["bare_ffi"][metric]
                            )
                        )
                        for metric in ("latency", "throughput")
                    },
                }
                payload["results"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(size, mode, row["paired_bound_vs_bare_ffi_us"], flush=True)
            # Independent higher-order check, outside the timing interval.
            hessian_product = jax.jit(
                lambda x, t: jax.jvp(jax.grad(lambda x: bound(x).sum()), (x,), (t,))[1]
            )
            np.testing.assert_allclose(
                hessian_product(*resident[0]),
                -np.sin(host[0][0]) * host[0][1],
                rtol=2e-6,
                atol=2e-6,
            )
            jb.synchronize(clear_cache=True)
        context.close()


if __name__ == "__main__":
    main()
