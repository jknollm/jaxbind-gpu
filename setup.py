from pathlib import Path

import jaxlib
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    ext_modules=[
        Pybind11Extension(
            "jaxbind_gpu._bridge",
            ["cpp/bridge.cc", "cpp/kernel_bridge.cc"],
            depends=[
                "cpp/graph_cache.h",
                "cpp/kernel_bridge.h",
                "cpp/inline_storage.h",
            ],
            include_dirs=[str(Path(jaxlib.__file__).parent / "include")],
            cxx_std=17,
            libraries=["dl"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
