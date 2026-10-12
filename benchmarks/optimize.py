"""Use an external GPU application in a compiled JAX optimization loop.

Choose a differentiable benchmark application with --app. This demonstration optimizes its
last floating/complex input, while holding other inputs fixed. The convolution
acceptance test also differentiates its image input separately.
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app",
        default="spectral_poisson",
        choices=(
            "spectral_poisson",
            "dense_solve",
            "sparse_diffusion",
            "image_blur",
            "torch_convolution",
            "nufft_synthesis",
            "reaction_diffusion",
        ),
    )
    parser.add_argument("--torch-path", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.torch_path:
        sys.path.insert(0, str(args.torch_path.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from applications.workloads import build
    import jaxbind_gpu as jb

    jax.config.update("jax_enable_x64", True)
    workload = build(args.app, "small")
    arrays = tuple(jax.device_put(x, jax.devices("gpu")[0]) for x in workload.host_args)

    def loss(parameter, fixed):
        value = workload.bound(*fixed, parameter)
        return jnp.real(jnp.vdot(value, value)) / (2 * value.size)

    value_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def optimize(parameter, fixed):
        def step(i, parameter):
            _, gradient = value_and_grad(parameter, fixed)
            # JAX complex gradients use the plain transpose convention.
            # Conjugating gives the steepest descent direction for a real loss.
            return parameter - jnp.asarray(0.05, parameter.dtype) * jnp.conj(gradient)

        return jax.lax.fori_loop(0, args.steps, step, parameter)

    fixed, parameter = arrays[:-1], arrays[-1]
    initial = jax.jit(loss)(parameter, fixed)
    with jax.transfer_guard_device_to_host("disallow"):
        fitted = optimize(parameter, fixed).block_until_ready()
    final = jax.jit(loss)(fitted, fixed)
    initial, final = float(initial), float(final)
    assert np.isfinite(final) and final < initial, (initial, final)
    print(f"{args.app}: {args.steps} compiled gradient steps on {fitted.device}")
    print(f"Loss: {initial:.8g} -> {final:.8g}")
    jb.synchronize(clear_cache=True)


if __name__ == "__main__":
    main()
