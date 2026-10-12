"""Shared timing, GPU exclusivity, and direct-library resource contracts."""

import os
import statistics
import subprocess
import time


class DirectContext:
    """Own reused plans; let native library allocators manage their temporaries.

    CuPy/Torch allocate and release temporary arrays on the same stream. Their
    pools can reuse that storage in stream order. Keeping every intermediate
    until the end of a timing batch would inflate the native memory baseline.
    Inputs and returned outputs remain owned by the benchmark caller.
    """

    def __init__(self, cp):
        self.device = cp.cuda.Device().id
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.cache = {}

    def cached(self, key, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        value = self.cache[key]
        self.retain(value)
        return value

    def retain(self, *values):
        pass

    def synchronize(self):
        self.stream.synchronize()

    def close(self):
        self.synchronize()
        self.cache.clear()


def check_exclusive(cp):
    properties = cp.cuda.runtime.getDeviceProperties(0)
    pci = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(pci, bytes):
        pci = pci.decode()
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"],
        text=True,
    )
    selected = None
    for row in rows.splitlines():
        bus, uuid = (field.strip() for field in row.split(","))
        if bus.lower().lstrip("0") == pci.lower().lstrip("0"):
            selected = uuid
    if selected is None:
        raise RuntimeError(f"Cannot identify selected GPU at {pci}")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    for row in rows.splitlines():
        uuid, pid = (field.strip() for field in row.split(","))
        if uuid == selected and int(pid) != os.getpid():
            raise RuntimeError(f"Benchmark GPU has another process: {pid}")
    return {"uuid": selected, "name": properties["name"].decode()}


def sample(function, argument_sets, synchronize, *, count, latency, warmup=True):
    # Exclusivity checks can leave the GPU and dispatch workers idle. Warm the
    # exact allocation/completion protocol immediately before each timed batch.
    if warmup:
        sample(
            function,
            argument_sets,
            synchronize,
            count=count,
            latency=latency,
            warmup=False,
        )
    outputs = []
    start = time.perf_counter()
    for i in range(count):
        outputs.append(function(*argument_sets[i % len(argument_sets)]))
        if latency:
            synchronize(outputs)
            outputs.clear()
    synchronize(outputs)
    elapsed = (time.perf_counter() - start) * 1e6 / count
    return elapsed


def measure_pair(variants, count=30, repeats=7):
    """Alternate warmed variants with the same output-retention protocol."""
    import cupy as cp
    import jax

    results = {}
    for name, function, argument_sets, *_ in variants:
        begin = time.perf_counter()
        function(*argument_sets[0]).block_until_ready()
        results[name] = dict(
            cold_call_us=(time.perf_counter() - begin) * 1e6,
            latency_samples_us=[],
            throughput_samples_us=[],
        )
        for inputs in argument_sets:
            function(*inputs).block_until_ready()
    with jax.transfer_guard_device_to_host("disallow"):
        for repetition in range(repeats):
            order = variants if repetition % 2 == 0 else variants[::-1]
            for name, function, argument_sets, *_ in order:
                check_exclusive(cp)
                for metric in ("latency", "throughput"):
                    results[name][f"{metric}_samples_us"].append(
                        sample(
                            function,
                            argument_sets,
                            jax.block_until_ready,
                            count=count,
                            latency=metric == "latency",
                        )
                    )
                check_exclusive(cp)
    for result in results.values():
        for metric in ("latency", "throughput"):
            result[f"{metric}_us"] = statistics.median(result[f"{metric}_samples_us"])
    return results
