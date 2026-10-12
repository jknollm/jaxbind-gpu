# Third-party notices

`src/jaxbind_gpu/jaxbind.py` derives from JAXbind's `jaxbind/jaxbind.py`.
`src/jaxbind_gpu/custom_map.py` copies JAXbind's `jaxbind/custom_map.py` under
its BSD-2-Clause option. Source notices and the upstream license are retained.

Source: https://github.com/NIFTy-PPL/JAXbind, version 1.3.1, revision
`22b0ff8860d99f6d622059ba925d981c0aca3277`.

The optional examples and benchmarks call installed CuPy, SciPy, PyTorch,
cuFINUFFT and NVIDIA Warp libraries. Those implementations are not copied or bundled.
The cuFINUFFT example reads the installed library's ABI and calls its C API.

Sources: https://cupy.dev/, https://scipy.org/, https://pytorch.org/,
https://github.com/flatironinstitute/finufft.
NVIDIA Warp: https://github.com/NVIDIA/warp.

The optional numerical reference is jax-finufft 1.3.1, revision
`8ae08bb11248b0a61c0c1e2ddc72e02741a5e7f7`. Its implementation is not bundled.
Benchmark artifacts identify the released reference and the benchmark variant
with plan reuse separately.

Source: https://github.com/flatironinstitute/jax-finufft.
