"""Execute ordinary CuPy/Torch functions against borrowed XLA outputs.

Only this module translates the array-function interface into write-to-output
callbacks. The runtime owns streams and lifetimes; jaxbind.py owns JAX rules.
"""

import jax

from ._runtime import current_context


def _invoke(function, inputs, targets, structure, kwargs, writes_output):
    if writes_output:
        outputs = jax.tree_util.tree_unflatten(structure, targets)
        result = function(*inputs, out=outputs, **kwargs)
        if result is None:
            return None, ()
    else:
        result = function(*inputs, **kwargs)
    values, actual = jax.tree_util.tree_flatten(result)
    if actual != structure:
        raise ValueError("External result structure differs from output specification")
    return result, values


def _validate(value, target, array_type):
    if not isinstance(value, array_type) or getattr(value, "is_cuda", True) is False:
        raise TypeError("External functions must return GPU arrays")
    if value.device != target.device:
        raise ValueError("External result is on a different GPU")
    if value.shape != target.shape or value.dtype != target.dtype:
        raise ValueError(
            "External result shape/dtype differs from output specification; declare output= for a different shape or dtype"
        )


def _check_alias(pointer, target_pointer, strides, target_strides):
    if pointer != target_pointer:
        return False
    if strides != target_strides:
        raise ValueError("Returned output aliases its target with different strides")
    return True


def _allocated_results(cp, function, inputs, outputs, structure, kwargs):
    """Lend each output to one matching allocation, then protect copy cycles."""
    available = list(outputs)
    allocator = cp.cuda.get_allocator()

    def allocate(size):
        for index, target in enumerate(available):
            if size and size == target.nbytes:
                return available.pop(index).data
        return allocator(size)

    with cp.cuda.using_allocator(allocate):
        result, values = _invoke(function, inputs, outputs, structure, kwargs, False)
    for target, value in zip(outputs, values):
        _validate(value, target, cp.ndarray)
    # A function can return a view of a borrowed allocation or reorder outputs.
    # Save every overlapping source before writing any destination. Allocation
    # forwarding is disabled here, so backups cannot overwrite another output.
    protected = []
    for target, value in zip(outputs, values):
        matches = value.data.ptr == target.data.ptr and value.strides == target.strides
        if matches:
            protected.append(value)
            continue
        memory = value.data.mem
        overlaps = any(
            memory.ptr < other.data.ptr + other.nbytes
            and other.data.ptr < memory.ptr + memory.size
            for other in outputs
        )
        protected.append(value.copy(order="C") if overlaps else value)
    return result, protected


def execute(
    function,
    inputs,
    outputs,
    structure,
    kwargs,
    backend,
    writes_output,
    *,
    allocate_output=False,
):
    """Queue external work, validate its results, and retain owned storage."""
    context = current_context()
    if backend == "cupy":
        import cupy as cp

        if allocate_output:
            result, values = _allocated_results(
                cp, function, inputs, outputs, structure, kwargs
            )
        else:
            result, values = _invoke(
                function, inputs, outputs, structure, kwargs, writes_output
            )
        for target, value in zip(outputs, values):
            if not allocate_output:
                _validate(value, target, cp.ndarray)
            if not _check_alias(
                value.data.ptr, target.data.ptr, value.strides, target.strides
            ):
                context.retain(value)
                cp.copyto(target, value)
        return

    import torch

    stream = context.cached(
        ("jaxbind_gpu", "torch_stream"),
        lambda: torch.cuda.ExternalStream(context.stream.ptr, device=context.device),
    )
    with (
        torch.cuda.device(context.device),
        torch.cuda.stream(stream),
        torch.no_grad(),
    ):
        # Both libraries already use the same stream. Exporting a capsule
        # avoids repeating device/consumer-stream negotiation for each array.
        inputs = tuple(
            torch.utils.dlpack.from_dlpack(x.__dlpack__(stream=-1)) for x in inputs
        )
        targets = tuple(
            torch.utils.dlpack.from_dlpack(x.__dlpack__(stream=-1)) for x in outputs
        )
        context.retain(stream, inputs, targets)
        result, values = _invoke(
            function, inputs, targets, structure, kwargs, writes_output
        )
        if result is not None:
            context.retain(result)
        for target, value in zip(targets, values):
            _validate(value, target, torch.Tensor)
            if not _check_alias(
                value.data_ptr(), target.data_ptr(), value.stride(), target.stride()
            ):
                target.copy_(value)
