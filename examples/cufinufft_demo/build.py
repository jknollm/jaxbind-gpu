"""Build the demo's native adapter against the installed public C ABI header."""

import os
from pathlib import Path
import shlex
import subprocess
import sysconfig

import pybind11
from jaxbind_gpu import include_dir


def main():
    folder = Path(__file__).resolve().parent
    output = folder / ("_bridge" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + include_dir(),
        str(folder / "native.cc"),
        str(folder / "module.cc"),
        "-ldl",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
