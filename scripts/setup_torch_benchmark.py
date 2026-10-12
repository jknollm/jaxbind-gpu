"""Install the tested optional Torch benchmark beside the JAX environment.

The isolated installation reuses the environment's existing CUDA libraries.
It leaves JAX dependencies in place; Torch wheel dependency pins may differ.
The convolution acceptance test checks this combination.
For a normal Torch installation, use its supported installation instructions
and omit --torch-path when running the application benchmarks.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path(".torch-benchmark"))
    args = parser.parse_args()
    site = Path(sysconfig.get_paths()["purelib"])
    cuda = site / "nvidia"
    if not cuda.is_dir():
        parser.error("Install this project's cuda12 extra first")
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    # One target installation avoids pip overwriting a shared nvidia namespace
    # directory when dependencies are installed in separate invocations.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(target),
            "--no-deps",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu129",
            "torch==2.9.1+cu129",
            "filelock",
            "sympy",
            "mpmath",
            "networkx",
            "jinja2",
            "MarkupSafe",
            "fsspec",
            "nvidia-cusparselt-cu12==0.7.1",
            "nvidia-curand-cu12==10.3.10.19",
            "nvidia-nvtx-cu12==12.9.79",
            "nvidia-cufile-cu12==1.14.1.1",
        ],
        check=True,
    )
    for directory in cuda.iterdir():
        link = target / "nvidia" / directory.name
        if directory.is_dir() and directory.name != "__pycache__" and not link.exists():
            link.symlink_to(directory, target_is_directory=True)
    print(f"Torch benchmark installed at {target}")
    print("Run benchmark_applications.py with --torch-path pointing to that directory.")


if __name__ == "__main__":
    main()
