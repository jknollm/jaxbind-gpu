import ctypes
from pathlib import Path
import struct
import subprocess
import uuid
import os
import sys

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import pytest

import jaxbind_gpu as jb


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    target = tmp_path_factory.mktemp("native") / "operations.so"
    subprocess.run(
        [
            "c++",
            "-O2",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-I" + jb.include_dir(),
            "-I" + str(Path(jaxlib.__file__).parent / "include"),
            str(Path(__file__).with_name("native_operations.cc")),
            "-ldl",
            "-o",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return ctypes.CDLL(str(target))


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.complex128, jnp.bfloat16, jnp.float8_e4m3fn]
)
@pytest.mark.parametrize("typed", [False, True])
def test_native_buffer_copy_and_transformations(gpu, native_library, dtype, typed):
    if typed:
        function = jb.NativeFunction(
            "test_copy_" + uuid.uuid4().hex,
            jax.ffi.pycapsule(native_library.typed_copy),
            owner=native_library,
        )
    else:
        function = jb.NativeFunction.from_cabi(
            native_library.copy_buffer, graph_compatible=True
        )

    def abstract(x):
        return ((x.shape, x.dtype),)

    op = jb.get_linear_call(function, function, abstract, abstract)
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4).astype(dtype)
    actual = jax.jit(jax.vmap(op))(x)[0]
    np.testing.assert_array_equal(
        jnp.real(actual).astype(jnp.float32), jnp.real(x).astype(jnp.float32)
    )
    if dtype == jnp.float32:
        grad = jax.jit(jax.grad(lambda x: jnp.sum(op(x)[0])))(x)
        np.testing.assert_array_equal(grad, np.ones(x.shape))


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.complex128, jnp.bfloat16, jnp.float8_e4m3fn]
)
def test_single_bind_factory_keeps_native_execution(gpu, native_library, dtype):
    function = jb.NativeFunction.from_cabi(native_library.copy_buffer)
    op = jb.bind(
        function,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        transpose=function,
    )
    x = jnp.arange(12, dtype=jnp.float32).astype(dtype)
    actual = jax.jit(op)(x)
    np.testing.assert_array_equal(jnp.real(actual).astype(jnp.float32), np.arange(12))
    if dtype == jnp.float32:
        gradient = jax.jit(jax.grad(lambda x: op(x).sum()))(x)
        np.testing.assert_array_equal(gradient, np.ones(12))
    hlo = jax.jit(op).lower(x).as_text()
    assert "jaxbind_gpu_pycall" not in hlo
    np.testing.assert_array_equal(
        jnp.real(jax.jit(jax.vmap(op))(x.reshape(3, 4))).astype(jnp.float32),
        np.arange(12).reshape(3, 4),
    )


def test_native_kernel_configuration_and_graph_loop(gpu, native_library):
    kernel = gpu.RawKernel(
        r"""
    extern "C" __global__ void scale(const float* x, float* out, long long n, float a) {
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = x[i] * a;
    }""",
        "scale",
    )
    kernel.compile()
    function = jb.NativeFunction.from_cabi(
        native_library.scale_buffer,
        encode=lambda factor: struct.pack("=Qf", kernel.kernel.ptr, factor),
        owner=kernel,
        graph_compatible=True,
    )

    def abstract(x, **kwargs):
        return ((x.shape, x.dtype),)

    op = jb.get_linear_call(function, function, abstract, abstract)
    loop = jax.jit(
        lambda x: jax.lax.fori_loop(0, 20, lambda i, x: op(x, factor=1.01)[0], x)
    )
    x = jnp.linspace(-1.0, 1.0, 513, dtype=jnp.float32)
    np.testing.assert_allclose(
        loop(x), np.asarray(x) * np.float32(1.01) ** 20, rtol=2e-6, atol=2e-6
    )
    grad = jax.jit(jax.grad(lambda x: jnp.sum(op(x, factor=2.0)[0])))(x)
    np.testing.assert_array_equal(grad, np.full(x.shape, 2.0))
    graph = jax.jit(
        lambda x: op(x, factor=1.01)[0],
        compiler_options={
            "xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL",
            "xla_gpu_graph_min_graph_size": 1,
        },
    )
    graph(x).block_until_ready()
    native_library.scale_count.restype = ctypes.c_uint64
    before = native_library.scale_count()
    for _ in range(4):
        graph(x).block_until_ready()
    assert native_library.scale_count() - before < 4


def test_native_errors_propagate(gpu, native_library):
    function = jb.NativeFunction.from_cabi(native_library.fail_buffer)
    op = jb.get_call(function, lambda x: ((x.shape, x.dtype),))
    with pytest.raises(Exception, match="Deliberate external native error"):
        jax.jit(op)(jnp.ones(4))[0].block_until_ready()


@pytest.mark.parametrize("typed", [False, True])
def test_existing_gpu_buffers_are_passed_directly(gpu, native_library, typed):
    if typed:
        function = jb.NativeFunction(
            "pointer_copy_" + uuid.uuid4().hex,
            jax.ffi.pycapsule(native_library.typed_copy),
            owner=native_library,
        )
    else:
        function = jb.NativeFunction.from_cabi(native_library.copy_buffer)
    op = jb.get_call(function, lambda x: ((x.shape, x.dtype),))
    external = gpu.arange(4096, dtype=gpu.float32)
    x = jax.dlpack.from_dlpack(external, copy=False)
    assert x.unsafe_buffer_pointer() == external.data.ptr
    with jax.transfer_guard_device_to_host("disallow"):
        y = jax.jit(op)(x)[0]
        y.block_until_ready()
    for name in ("last_input_pointer", "last_output_pointer"):
        getattr(native_library, name).restype = ctypes.c_uint64
    assert native_library.last_input_pointer() == external.data.ptr
    assert native_library.last_output_pointer() == y.unsafe_buffer_pointer()
    external_output = gpu.from_dlpack(y)
    assert external_output.data.ptr == y.unsafe_buffer_pointer()
    gpu.testing.assert_array_equal(external_output, external)


def test_named_ffi_export_roundtrip_in_new_process(gpu, native_library, tmp_path):
    import jax.export

    name = "jaxbind_gpu_export_copy_" + uuid.uuid4().hex
    function = jb.NativeFunction(
        name, jax.ffi.pycapsule(native_library.typed_copy), owner=native_library
    )
    op = jb.get_call(function, lambda x: ((x.shape, x.dtype),))
    exported = jax.export.export(
        jax.jit(op),
        platforms=("cuda",),
        disabled_checks=(jax.export.DisabledSafetyCheck.custom_call(name),),
    )(jax.ShapeDtypeStruct((8,), np.float32))
    artifact = tmp_path / "operation.jax"
    artifact.write_bytes(exported.serialize())
    code = """
import ctypes, sys
import jax, jax.export, jax.numpy as jnp
import numpy as np
from pathlib import Path
import jaxbind_gpu as jb
library = ctypes.CDLL(sys.argv[1])
jb.NativeFunction(sys.argv[2], jax.ffi.pycapsule(library.typed_copy), owner=library)
exported = jax.export.deserialize(Path(sys.argv[3]).read_bytes())
result = exported.call(jnp.arange(8, dtype=jnp.float32))[0]
np.testing.assert_array_equal(result, np.arange(8))
native = jb.NativeFunction.from_cabi(library.copy_buffer)
bound = jb.get_call(native, lambda x: ((x.shape, x.dtype),))
np.testing.assert_array_equal(jax.jit(bound)(result)[0], np.arange(8))
assert "cupy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, native_library._name, name, str(artifact)],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        timeout=60,
    )


@pytest.mark.parametrize("configure_binding", [False, True])
def test_native_side_effect_and_alias_options(gpu, native_library, configure_binding):
    options = {"has_side_effect": True}
    function = jb.NativeFunction.from_cabi(
        native_library.effect_buffer, **({} if configure_binding else options)
    )
    effect = jb.get_call(
        function, lambda x: (), **(options if configure_binding else {})
    )
    native_library.effect_count.restype = ctypes.c_uint64
    before = native_library.effect_count()

    @jax.jit
    def execute(x):
        effect(x)
        effect(x)
        return x + 1

    execute(jnp.arange(4.0)).block_until_ready()
    jax.effects_barrier()
    assert native_library.effect_count() - before == 2
    options = {"input_output_aliases": {0: 0}}
    function = jb.NativeFunction.from_cabi(
        native_library.alias_buffer, **({} if configure_binding else options)
    )
    alias = jb.get_call(
        function,
        lambda x: ((x.shape, x.dtype),),
        **(options if configure_binding else {}),
    )
    result = jax.jit(alias, donate_argnums=(0,))(jnp.arange(8.0))[0]
    np.testing.assert_array_equal(result, np.arange(8.0))
