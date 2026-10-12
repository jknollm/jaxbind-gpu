# Build and check distributions

Install the release tools in a development environment:

```sh
python -m pip install '.[test,release]'
python scripts/check_package.py --release
```

The check builds a source archive and builds its wheel from that archive, using
isolated environments with the declared build dependencies. `--release` audits
the extension, creates a manylinux wheel and checks distribution metadata and
the package description with Twine. It then installs that wheel into a fresh
CPU environment without CuPy, Torch, cuFINUFFT or Warp and runs the core CPU tests.
No command uploads a distribution.

Outputs go to `dist/` and `validation-build.json`. The source archive contains
documentation, examples, tests and benchmark tools. Generated measurement files
and validation records are excluded. The wheel
contains only the runtime package, native bridge, C ABI header and license notices.
To validate a wheel already built:

```sh
python scripts/check_package.py --wheel dist/PACKAGE.whl
python -m twine check --strict dist/PACKAGE.whl dist/PACKAGE.tar.gz
```

Distribute the manylinux wheel and source archive. The check removes its plain
`linux_*` intermediate wheel after successful release validation. Auditwheel
determines the manylinux compatibility floor
from the binary's symbols and dependencies. On the tested build host, the wheel
targets CPython 3.12 and x86_64 with glibc 2.34 or newer. Other Python versions or
architectures need their own build and validation. Build on an older compatible
Linux image to support an older glibc floor.

The CUDA driver is loaded at runtime and is not bundled. An ABI audit does not
test driver compatibility or libraries loaded through `dlopen`. Run the GPU suite
and example checks on the installed release wheel before distributing it:

```sh
python -m pip install --force-reinstall --no-deps dist/PACKAGE.whl
python scripts/test_gpu.py -q
python scripts/run_gpu.py scripts/check_examples.py
```

Package metadata uses `docs/package.md` as the index description so its links do
not depend on a repository URL. `README.md` links to the documentation in the
source archive. Keep installation instructions and compatibility claims aligned.

The license is BSD-2-Clause. Retain both copyright lines and the source notices
on code derived from JAXbind. Optional application libraries are not bundled.

See the [platform tag specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)
and [Auditwheel documentation](https://github.com/pypa/auditwheel) for Linux wheel
compatibility rules.
