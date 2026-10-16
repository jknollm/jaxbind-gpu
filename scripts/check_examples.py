"""Run the self-checking integration lessons sequentially on the selected GPU.

Use scripts/run_gpu.py to select an idle GPU first. Optional integrations run
when their dependencies are installed. --torch-path selects an isolated installation.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-path", type=Path)
    args = parser.parse_args()
    env = dict(os.environ)
    if args.torch_path:
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(args.torch_path.resolve()), env.get("PYTHONPATH")))
        )
    examples = sorted((ROOT / "examples").glob("*.py"))
    passed, skipped = 0, 0
    for example in examples:
        dependency = {
            "torch_function": "torch",
            "nufft_native": "cufinufft",
            "nufft_python": "cufinufft",
            "image_filter": "scipy",
            "warp_sparse": "warp",
            "shtns_python": "shtns",
        }.get(example.stem)
        if (
            dependency
            and not (dependency == "torch" and args.torch_path)
            and importlib.util.find_spec(dependency) is None
        ):
            print(f"Skipping {example.name}: install {dependency}", flush=True)
            skipped += 1
            continue
        if (
            example.stem == "shtns_python"
            and importlib.util.find_spec("_shtns_cuda") is None
        ):
            print("Skipping shtns_python.py: build SHTns with CUDA support", flush=True)
            skipped += 1
            continue
        if example.stem == "nufft_native" and not list(
            (ROOT / "examples/cufinufft_demo").glob("_bridge*.so")
        ):
            print(
                "Skipping nufft_native.py: run python examples/cufinufft_demo/build.py",
                flush=True,
            )
            skipped += 1
            continue
        print(f"Checking {example.name}", flush=True)
        subprocess.run([sys.executable, str(example)], env=env, check=True, cwd=ROOT)
        passed += 1
    print(f"{passed} examples passed, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
