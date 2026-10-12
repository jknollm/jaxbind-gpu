"""Small real-library adapters; no host copies of application input arrays.

Each ``run`` also supplies the direct-library benchmark. This keeps algorithms,
output copies, resource reuse, and options identical on both execution paths.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxbind_gpu as jb


def same_shape(x):
    return ((x.shape, x.dtype),)


def leading_batch(call, args, axes):
    return call(jnp.moveaxis(args[0], axes[0], 0)), (0,)


class LinearApplication:
    """Supply shape inference and a real transpose for a linear operation."""

    fixed_args = 0
    batching_rule = None
    workspace_bytes = 0

    def __init__(self):
        self.observed = None
        self.enable_graph(False)

    def enable_graph(self, enabled=True):
        self.op = jb.get_linear_call(
            self._forward,
            self._transpose,
            self.abstract,
            self.abstract,
            first_n_args_fixed=self.fixed_args,
            batching_rule=self.batching_rule,
            graph_compatible=enabled,
            workspace_bytes=self.workspace_bytes,
        )

    def __call__(self, *args):
        return self.op(*args)[0]

    def abstract(self, *args):
        return same_shape(args[-1])

    def _forward(self, out, args, config):
        context = jb.current_context()
        self.observed = {
            "inputs": tuple(x.data.ptr for x in args),
            "output": out[0].data.ptr,
            "stream": context.stream.ptr,
            "device": context.device,
        }
        self.run(out[0], args, context)

    def _transpose(self, out, args, config):
        self.run(out[0], args, jb.current_context(), transpose=True)


class SpectralPoisson(LinearApplication):
    """Solve (1 - discrete periodic Laplacian) u = f with cached cuFFT plans."""

    batching_rule = staticmethod(leading_batch)

    def run(self, out, args, context, *, transpose=False):
        import cupy as cp
        from cupyx.scipy.fft import fftn, get_fft_plan, ifftn

        x = args[0]
        work = x.astype(np.complex64 if x.dtype == np.float32 else np.complex128)

        def resources():
            h, w = x.shape[-2:]
            fy = cp.fft.fftfreq(h).astype(x.dtype)
            fx = cp.fft.fftfreq(w).astype(x.dtype)
            multiplier = 1 / (
                1
                + 4 * cp.sin(cp.pi * fy[:, None]) ** 2
                + 4 * cp.sin(cp.pi * fx[None, :]) ** 2
            )
            return get_fft_plan(work, axes=(-2, -1)), multiplier

        plan, multiplier = context.cached((x.shape, x.dtype.str), resources)
        spectrum = fftn(work, axes=(-2, -1), plan=plan)
        spectrum *= multiplier
        value = ifftn(spectrum, axes=(-2, -1), plan=plan)
        cp.copyto(out, value.real)
        context.retain(work, spectrum, value)


class DenseSolve(LinearApplication):
    """Solve a dense system with cuSOLVER; matrix is fixed for differentiation."""

    fixed_args = 1

    def run(self, out, args, context, *, transpose=False):
        import cupy as cp

        matrix, rhs = args
        value = cp.linalg.solve(matrix.T if transpose else matrix, rhs)
        cp.copyto(out, value)
        context.retain(value)


class SparseDiffusion(LinearApplication):
    """Apply a sorted CSR diffusion operator with cuSPARSE to resident buffers."""

    fixed_args = 3

    def run(self, out, args, context, *, transpose=False):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix

        data, indices, indptr, x = args
        n = x.shape[0]
        matrix = csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
        # The application constructs canonical sorted CSR. Avoid an unnecessary
        # device query of that already known invariant on each application call.
        matrix.has_sorted_indices = True
        matrix.has_canonical_format = True
        value = (matrix.T if transpose else matrix).dot(x)
        cp.copyto(out, value)
        context.retain(matrix, value)


class ImageBlur(LinearApplication):
    """Periodic symmetric image convolution with cupyx.scipy.ndimage."""

    batching_rule = staticmethod(leading_batch)

    def run(self, out, args, context, *, transpose=False):
        import cupy as cp
        from cupyx.scipy.ndimage import convolve

        x = args[0]

        def weights():
            line = cp.asarray([1, 2, 1], dtype=x.dtype)
            return (line[:, None] * line[None, :] / 16).reshape(
                (1,) * (x.ndim - 2) + (3, 3)
            )

        kernel = context.cached((x.ndim, x.dtype.str), weights)
        convolve(x, kernel, output=out, mode="wrap")


class ArrayApplication:
    """Exercise the public factory with a GPU function and optional out support."""

    workspace_bytes = 0

    def __init__(self, name, output_shape, dtype):
        import cupy as cp
        from cupyx.scipy.special import erf

        self.name = name
        self.writes_output = name != "sort"
        self.spec = jax.ShapeDtypeStruct(output_shape, dtype)
        self.function = {
            "row_reduction": lambda x, *, out=None: cp.sum(x, axis=-1, out=out),
            "prefix_scan": lambda x, *, out=None: cp.cumsum(x, out=out),
            "batched_gemm": lambda a, b, *, out=None: cp.matmul(a, b, out=out),
            "special_erf": lambda x, *, out=None: erf(x, out=out),
            "sort": cp.sort,
        }[name]
        self.enable_graph(False)

    def enable_graph(self, enabled=True):
        self.op = jb.bind(
            self.function,
            output=self.spec,
            out=self.writes_output,
            graph_compatible=enabled,
            workspace_bytes=self.workspace_bytes,
        )

    def __call__(self, *args):
        return self.op(*args)

    def run(self, out, args, context):
        import cupy as cp

        if self.name == "sort":
            value = self.function(*args)
            cp.copyto(out, value)
            context.retain(value)
        else:
            self.function(*args, out=out)


class ReactionDiffusion:
    """A fused external CUDA stencil, launched directly from XLA buffers."""

    def __init__(self):
        import cupy as cp

        self.kernel = cp.RawKernel(
            r"""
            extern "C" __global__ void reaction(const float* x, float* out, int h, int w) {
              int i = blockIdx.x * blockDim.x + threadIdx.x;
              if (i >= h * w) return;
              int r = i / w, c = i % w;
              float v = x[i];
              float neighbors = x[((r + h - 1) % h) * w + c]
                  + x[((r + 1) % h) * w + c]
                  + x[r * w + (c + w - 1) % w]
                  + x[r * w + (c + 1) % w];
              out[i] = .5f * v + .125f * neighbors + .01f * v * (1.f - v * v);
            }
            """,
            "reaction",
        )
        self.op = jb.bind(
            self.kernel,
            output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            launch=(lambda x: ((x.size + 255) // 256,), (256,)),
            scalars=lambda x: (np.int32(x.shape[0]), np.int32(x.shape[1])),
            jvp=self.jvp,
        )

    @staticmethod
    def jvp(primals, tangents):
        x, dx = primals[0], tangents[0]
        neighbors = sum(
            jnp.roll(dx, shift, axis=axis) for axis in (0, 1) for shift in (-1, 1)
        )
        return 0.5 * dx + 0.125 * neighbors + 0.01 * (1 - 3 * x * x) * dx

    def __call__(self, x):
        return self.op(x)

    def enable_graph(self, enabled=True):
        # Native CUDA kernel bindings support capture without Python callbacks.
        pass

    def run(self, out, args, context):
        x = args[0]
        self.kernel(
            ((x.size + 255) // 256,),
            (256,),
            (x, out, np.int32(x.shape[0]), np.int32(x.shape[1])),
        )


class TorchConvolution:
    """Trainable NCHW convolution using Torch/cuDNN forward and both adjoints.

    DLPack borrows the input and XLA output storage. Torch computes a temporary
    result because its convolution API has no output parameter. Both benchmark
    paths include the same device-to-device output copy.
    """

    def __init__(self, input_shape, weight_shape):
        import torch

        self.torch = torch
        self.input_shape = tuple(input_shape)
        self.weight_shape = tuple(weight_shape)
        self.observed = None
        # Explicit full FP32 for comparison against the independent JAX oracle.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.op = jb.get_linear_call(
            self._forward,
            (self._input_adjoint, self._weight_adjoint),
            self.abstract,
            (self.abstract_input, self.abstract_weight),
        )

    def __call__(self, x, weights):
        return self.op(x, weights)[0]

    def abstract(self, x, weights):
        if x.shape != self.input_shape or weights.shape != self.weight_shape:
            raise ValueError("Construct this adapter for the supplied NCHW/OIHW shapes")
        if x.dtype != weights.dtype or x.dtype != np.dtype("float32"):
            raise TypeError("The convolution application uses float32")
        n, _, h, w = x.shape
        o, _, kh, kw = weights.shape
        if kh != 3 or kw != 3:
            raise ValueError("The convolution application uses 3 x 3 kernels")
        return (((n, o, h, w), x.dtype),)

    def abstract_input(self, weights, cotangent):
        return ((self.input_shape, cotangent.dtype),)

    def abstract_weight(self, x, cotangent):
        return ((self.weight_shape, cotangent.dtype),)

    def _forward(self, out, args, config):
        self.run(out[0], args, jb.current_context())

    def _input_adjoint(self, out, args, config):
        self.run(out[0], args, jb.current_context(), adjoint="input")

    def _weight_adjoint(self, out, args, config):
        self.run(out[0], args, jb.current_context(), adjoint="weight")

    def run(self, out, args, context, *, adjoint=None):
        torch = self.torch
        stream = torch.cuda.ExternalStream(context.stream.ptr, device=context.device)
        with (
            torch.cuda.device(context.device),
            torch.cuda.stream(stream),
            torch.no_grad(),
        ):
            tensors = tuple(torch.utils.dlpack.from_dlpack(x) for x in args)
            target = torch.utils.dlpack.from_dlpack(out)
            if adjoint == "input":
                value = torch.nn.grad.conv2d_input(
                    self.input_shape, tensors[0], tensors[1], padding=1
                )
            elif adjoint == "weight":
                value = torch.nn.grad.conv2d_weight(
                    tensors[0], self.weight_shape, tensors[1], padding=1
                )
            else:
                value = torch.nn.functional.conv2d(*tensors, padding=1)
                self.observed = {
                    "inputs": tuple(t.data_ptr() for t in tensors),
                    "output": target.data_ptr(),
                    "stream": torch.cuda.current_stream().cuda_stream,
                    "device": context.device,
                }
            target.copy_(value)
            context.retain(stream, *tensors, target, value)


def torch_reference(x, weights):
    return jax.lax.conv_general_dilated(
        x,
        weights,
        (1, 1),
        "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=jax.lax.Precision.HIGHEST,
    )
