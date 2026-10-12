"""Build distributions, install the wheel in a fresh CPU environment and test it.

Run from a development environment with the release tools installed.
The temporary environment installs only core dependencies and test tools.
The default checks need no CUDA device and can run in CI.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, help="Validate an existing wheel without rebuilding"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Audit and repair the Linux wheel, then check distribution metadata",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "validation-build.json")
    args = parser.parse_args()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    # A caller's reference/Torch imports must not enter the fresh environment.
    env.pop("PYTHONPATH", None)
    # Release utilities must use the caller's environment without activation.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    checks = {}
    sdist = None
    if args.wheel:
        wheel = args.wheel.resolve()
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                str(ROOT / "dist"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        wheel = max((ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime_ns)
        sdist = max(
            (ROOT / "dist").glob("*.tar.gz"), key=lambda p: p.stat().st_mtime_ns
        )
        checks["isolated_source_build"] = "passed"
    built_wheel = wheel
    if args.release:
        subprocess.run(
            [sys.executable, "-m", "auditwheel", "show", str(wheel)],
            cwd=ROOT,
            env=env,
            check=True,
        )
        if "manylinux" not in wheel.name:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "auditwheel",
                    "repair",
                    "--wheel-dir",
                    str(ROOT / "dist"),
                    str(wheel),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            wheel = max(
                (ROOT / "dist").glob("*manylinux*.whl"),
                key=lambda p: p.stat().st_mtime_ns,
            )
        subprocess.run(
            [sys.executable, "-m", "twine", "check", "--strict", str(wheel)]
            + ([str(sdist)] if sdist else []),
            cwd=ROOT,
            env=env,
            check=True,
        )
        checks.update(linux_abi_audit="passed", distribution_metadata="passed")
    with tempfile.TemporaryDirectory(prefix="jaxbind-gpu-check-") as temporary:
        directory = Path(temporary)
        environment = directory / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin/python"
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel) + "[test]"],
            cwd=directory,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"], cwd=directory, env=env, check=True
        )
        probe = (
            "import importlib.util, json, jax, jaxbind_gpu as jb; "
            "assert jax.default_backend() == 'cpu'; "
            "assert all(importlib.util.find_spec(name) is None for name in "
            "('cupy', 'torch', 'cufinufft', 'warp')); "
            "print(json.dumps(dict(version=jb.__version__, path=jb.__file__, "
            "jax=jax.__version__, backend=jax.default_backend())))"
        )
        result = subprocess.run(
            [str(python), "-c", probe],
            cwd=directory,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        installed = json.loads(result.stdout)
        installed["path"] = str(Path(installed["path"]).relative_to(environment))
        # The core suite excludes optional application/library integrations.
        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                str(ROOT / "tests"),
                "--ignore=" + str(ROOT / "tests/demos"),
                "--ignore=" + str(ROOT / "tests/test_applications.py"),
                "-q",
            ],
            cwd=directory,
            env=env,
            check=True,
        )
        with zipfile.ZipFile(wheel) as archive:
            payload = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith("jaxbind_gpu/") and not name.endswith("/")
            }
        record = {
            "wheel": str(wheel.relative_to(ROOT)),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "installed": installed,
            "payload_sha256": payload,
            "checks": {
                **checks,
                "isolated_install": "passed",
                "optional_libraries_absent": "passed",
                "cpu_import": "passed",
                "core_cpu_tests": "passed",
                "pip_check": "passed",
            },
        }
        if sdist:
            record["source_archive"] = str(sdist.relative_to(ROOT))
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.release and sdist and built_wheel != wheel:
        built_wheel.unlink()
    print(f"Package checks passed. Saved {args.output}")


if __name__ == "__main__":
    main()
