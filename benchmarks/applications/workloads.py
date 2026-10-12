"""Application definitions, size presets, and independent acceptance oracles."""

from dataclasses import dataclass

import jax
import numpy as np
from scipy import sparse

from .operators import (
    ArrayApplication,
    ReactionDiffusion,
    DenseSolve,
    ImageBlur,
    SparseDiffusion,
    SpectralPoisson,
    TorchConvolution,
    torch_reference,
)


APPLICATIONS = {
    "warp_sparse": "Three-component block-sparse product using NVIDIA Warp",
    "spectral_poisson": "Periodic screened Poisson equation using CuPy/cuFFT",
    "dense_solve": "Dense multi-RHS linear system using CuPy/cuSOLVER",
    "sparse_diffusion": "Eight CSR diffusion steps using cupyx/cuSPARSE",
    "image_blur": "Periodic image filtering using cupyx.scipy.ndimage",
    "torch_convolution": "Trainable image convolution using PyTorch/cuDNN",
    "nufft_synthesis": "Irregular Fourier sampling using native cuFINUFFT",
    "row_reduction": "Row sums using CuPy reduction kernels",
    "prefix_scan": "Inclusive prefix sums using CuPy scan kernels",
    "batched_gemm": "Batched dense products using CuPy/cuBLAS",
    "special_erf": "Gaussian error function using cupyx.scipy.special",
    "sort": "Sorting GPU arrays using CuPy/Thrust",
    "reaction_diffusion": "Fused external CUDA reaction/diffusion stencil through native kernel binding",
}


def builtin_reference(name):
    """Equivalent mathematics for context; external algorithms stay unchanged."""
    import jax.numpy as jnp
    from jax.scipy.special import erf

    def reaction(x):
        neighbors = sum(
            jnp.roll(x, shift, axis=axis) for axis in (0, 1) for shift in (-1, 1)
        )
        return 0.5 * x + 0.125 * neighbors + 0.01 * x * (1 - x * x)

    references = {
        "row_reduction": lambda x: jnp.sum(x, axis=-1),
        "prefix_scan": jnp.cumsum,
        "batched_gemm": lambda a, b: jnp.matmul(
            a, b, precision=jax.lax.Precision.HIGHEST
        ),
        "special_erf": erf,
        "sort": jnp.sort,
        "reaction_diffusion": reaction,
    }
    if name not in references:
        raise ValueError(f"No JAX builtin comparison configured for {name}")
    return references[name]


@dataclass
class Workload:
    name: str
    size: str
    operator: object
    host_args: tuple
    reference: object
    backend: str = "Python GPU callback"
    steps: int = 1
    comparison_indices: object = None
    rtol: float = 3e-5
    atol: float = 3e-6
    output_spec: object = None

    def bound(self, *args):
        if self.steps == 1:
            return self.operator(*args)
        fixed, x = args[:-1], args[-1]
        return jax.lax.fori_loop(
            0, self.steps, lambda i, x: self.operator(*fixed, x), x
        )

    def direct(self, args, context):
        import cupy as cp

        if self.output_spec is None:
            self.output_spec = jax.eval_shape(
                self.operator,
                *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args),
            )
        spec = self.output_spec
        # Allocate one output per external call, just as XLA does for these
        # non-aliasing operations. Keep all queued allocations alive until sync.
        for _ in range(self.steps):
            output = cp.empty(spec.shape, dtype=spec.dtype)
            self.operator.run(output, args, context)
            context.retain(output)
            args = (*args[:-1], output)
        return output

    def argument_sets(self):
        first = self.host_args
        last = first[-1]
        # Change data only; topology, matrix, and geometry stay resident. The
        # second call cannot be satisfied by a cached output from the first.
        if self.name == "torch_convolution":
            second = (first[0] * np.float32(1.1), last * np.float32(0.9))
        else:
            second = (*first[:-1], last * np.float32(1.1))
        return first, second


class NufftSynthesis:
    def __call__(self, coordinates, coefficients):
        from examples.cufinufft_demo import nufft2

        return nufft2(coordinates, coefficients, eps=1e-10, upsampfac=2.0)

    def run(self, out, args, context):
        import cufinufft

        coordinates, coefficients = args
        plan = context.cached(
            (coefficients.shape, coefficients.dtype.str),
            lambda: cufinufft.Plan(
                2,
                coefficients.shape,
                eps=1e-10,
                dtype=coefficients.dtype,
                isign=-1,
                upsampfac=2.0,
                gpu_stream=context.stream.ptr,
                gpu_device_id=context.device,
            ),
        )
        plan.setpts(coordinates)
        plan.execute(coefficients, out=out)


def build(name, size, *, seed=820):
    if name not in APPLICATIONS or size not in ("small", "medium"):
        raise ValueError("Select a listed application and a small/medium size")
    rng = np.random.default_rng(seed)
    small = size == "small"
    if name == "warp_sparse":
        from .warp_sparse import build as build_warp

        return build_warp(size, rng)
    if name == "reaction_diffusion":
        shape = (64, 64) if small else (1024, 1024)
        x = rng.uniform(-1, 1, shape).astype("float32")

        def reference(x):
            v = x.astype("float64")
            neighbors = sum(
                np.roll(v, shift, axis=axis) for axis in (0, 1) for shift in (-1, 1)
            )
            return 0.5 * v + 0.125 * neighbors + 0.01 * v * (1 - v * v)

        return Workload(
            name,
            size,
            ReactionDiffusion(),
            (x,),
            reference,
            backend="Native CUDA kernel",
        )
    if name in ("row_reduction", "prefix_scan", "batched_gemm", "special_erf", "sort"):
        from scipy.special import erf

        if name == "batched_gemm":
            shape = (4, 32, 32) if small else (8, 256, 256)
            args = tuple(rng.normal(size=shape).astype("float32") for _ in range(2))
            output_shape = shape

            def reference(a, b):
                return a.astype("float64") @ b.astype("float64")
        else:
            shape = (
                ((64, 128) if small else (4096, 1024))
                if name == "row_reduction"
                else ((4096,) if small else (1048576,))
            )
            # Positive scan inputs make the oracle's relative error meaningful.
            x = (
                rng.uniform(size=shape)
                if name == "prefix_scan"
                else rng.normal(size=shape)
            ).astype("float32")
            args = (x,)
            output_shape = shape[:-1] if name == "row_reduction" else shape
            reference = {
                "row_reduction": lambda x: np.sum(x.astype("float64"), axis=-1),
                "prefix_scan": lambda x: np.cumsum(x.astype("float64")),
                "special_erf": erf,
                "sort": np.sort,
            }[name]
        return Workload(
            name,
            size,
            ArrayApplication(name, output_shape, np.float32),
            args,
            reference,
            rtol=3e-5,
            atol=3e-4 if name in ("row_reduction", "batched_gemm") else 3e-6,
        )
    if name == "spectral_poisson":
        shape = (64, 64) if small else (1024, 1024)
        x = rng.normal(size=shape).astype("float32")
        fy, fx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
        multiplier = 1 / (
            1
            + 4 * np.sin(np.pi * fy[:, None]) ** 2
            + 4 * np.sin(np.pi * fx[None, :]) ** 2
        )

        def reference(x):
            return np.fft.ifft2(np.fft.fft2(x) * multiplier).real

        return Workload(name, size, SpectralPoisson(), (x,), reference)
    if name == "image_blur":
        shape = (64, 64) if small else (1024, 1024)
        x = rng.normal(size=shape).astype("float32")

        def reference(x):
            value = np.zeros_like(x)
            for i, a in enumerate((1, 2, 1)):
                for j, b in enumerate((1, 2, 1)):
                    value += (a * b / 16) * np.roll(x, (i - 1, j - 1), axis=(-2, -1))
            return value

        return Workload(name, size, ImageBlur(), (x,), reference)
    if name == "dense_solve":
        n, k = (64, 4) if small else (512, 16)
        # A well-conditioned system avoids comparing unstable numerical results.
        a = rng.normal(size=(n, n)).astype("float32") / (4 * np.sqrt(n)) + np.eye(
            n, dtype="float32"
        )
        b = rng.normal(size=(n, k)).astype("float32")

        def reference(a, b):
            return np.linalg.solve(a.astype("float64"), b.astype("float64"))

        return Workload(name, size, DenseSolve(), (a.astype("float32"), b), reference)
    if name == "sparse_diffusion":
        n = 4096 if small else 262144
        matrix = sparse.diags(
            (
                np.full(n - 1, 0.2, dtype="float32"),
                np.full(n, 0.5, dtype="float32"),
                np.full(n - 1, 0.2, dtype="float32"),
            ),
            (-1, 0, 1),
            format="csr",
        )
        matrix.sort_indices()
        x = rng.normal(size=n).astype("float32")
        args = (
            matrix.data,
            matrix.indices.astype("int32"),
            matrix.indptr.astype("int32"),
            x,
        )

        def reference(data, indices, indptr, x):
            a = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
            for _ in range(8):
                x = a @ x
            return x

        return Workload(name, size, SparseDiffusion(), args, reference, steps=8)
    if name == "torch_convolution":
        shape = (1, 4, 32, 32) if small else (4, 16, 128, 128)
        weights_shape = (8, 4, 3, 3) if small else (32, 16, 3, 3)
        x = rng.normal(size=shape).astype("float32")
        weights = rng.normal(size=weights_shape).astype("float32") / np.sqrt(
            np.prod(weights_shape[1:])
        )
        return Workload(
            name,
            size,
            TorchConvolution(shape, weights_shape),
            (x, weights.astype("float32")),
            torch_reference,
        )
    n = 4096 if small else 65536
    coordinates = rng.uniform(-np.pi, np.pi, n).astype("float64")
    coefficients = ((rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(n)).astype(
        "complex128"
    )
    selected = np.arange(min(16, n))
    modes = np.arange(n) - n // 2

    def reference(coordinates, coefficients):
        # Independent dense sums at selected points bound oracle memory to
        # about 16 MB for the larger workload, rather than creating an N^2 array.
        return np.exp(
            -1j * coordinates[selected, None].astype("float64") * modes
        ) @ coefficients.astype("complex128")

    return Workload(
        name,
        size,
        NufftSynthesis(),
        (coordinates, coefficients),
        reference,
        backend="Native C ABI",
        comparison_indices=selected,
        rtol=1e-8,
        atol=1e-8,
    )
