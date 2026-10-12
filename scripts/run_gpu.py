"""Run a Python program on one idle GPU, optionally waiting for availability."""

import argparse
import os
import subprocess
import sys
import time

from test_gpu import idle_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", type=float, default=0, help="Maximum seconds to wait")
    parser.add_argument("--devices", type=int, default=1, help="Number of idle GPUs")
    parser.add_argument("program", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.devices < 1:
        parser.error("--devices must be positive")
    if args.program and args.program[0] == "--":
        args.program.pop(0)
    if not args.program:
        parser.error("Supply a Python script and its arguments")
    deadline = time.monotonic() + args.wait
    while True:
        try:
            device = ",".join(idle_devices(args.devices))
            break
        except SystemExit:
            if time.monotonic() >= deadline:
                raise
            print("Waiting for an idle GPU", flush=True)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    print(f"Running on idle device {device}", flush=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    return subprocess.call([sys.executable, *args.program], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
