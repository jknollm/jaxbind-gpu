"""Render the current overview as runtime, memory and comparison tables."""

import csv
import json
import statistics

import numpy as np


def write_report(overview, directory):
    lines = [
        "# Current memory and runtime comparison",
        "",
        f"Version {overview['version']} on {overview['hardware']}. "
        "Tables derive from the current raw evidence. Rebuild with "
        "`python scripts/summarize_benchmarks.py`.",
        "",
        "Time cells give blocked latency / batch throughput in microseconds per "
        "call. Inputs change on the GPU; each throughput call returns a fresh "
        "result held until completion. Compilation and first execution are "
        "excluded. Variant order alternates and each timed batch has an immediate "
        "warm batch. GPU exclusivity and device-to-host transfer guards apply.",
        "",
        "Memory tables use MiB; CSV fields use bytes. Fresh-process memory runs "
        "state their held-output counts. Base application and wrapper sweeps "
        "hold four outputs; recording and Warp controls hold 32. Driver sampling includes contexts, modules, pools "
        "and library-private allocations but can miss short peaks. "
        "Separate "
        "allocator peaks need not coincide. Do not add them as a total. "
        "Compiler memory is distinct from measured allocator memory.",
        "",
    ]
    timing_rows, memory_rows = [], []

    def table(title, headers, rows):
        lines.extend(
            [
                f"## {title}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        lines.append("")

    def source(name):
        lines.extend([f"Source: [current-{name}](current-{name}.json).", ""])

    def read(name):
        return json.loads((directory / f"current-{name}.json").read_text())

    def times(values):
        if values is None:
            return "-"
        return f"{values['latency_us']:.1f} / {values['throughput_us']:.1f}"

    def mib(value):
        return "-" if value is None else f"{value / 2**20:.3f}"

    def runtime(section, configuration, values):
        for variant, timing in values.items():
            timing_rows.append(
                {
                    "section": section,
                    "configuration": configuration,
                    "variant": variant,
                    "latency_us": timing["latency_us"],
                    "throughput_us": timing["throughput_us"],
                }
            )

    interfaces = []
    for row in read("interfaces")["applications"]:
        values = row["results"]
        interfaces.append(
            [
                row["application"],
                row["size"],
                *[
                    times(values.get(name))
                    for name in (
                        "native",
                        "eager_copy",
                        "jit_copy",
                        "eager_out",
                        "jit_out",
                        "cached_out",
                        "jit_allocate",
                    )
                ],
            ]
        )
        runtime("interfaces", f"{row['application']}/{row['size']}", values)
    table(
        "Public factory execution",
        [
            "Operation",
            "Elements",
            "Native",
            "Direct copy",
            "JIT copy",
            "Direct out",
            "JIT out",
            "Cached out",
            "JIT allocate",
        ],
        interfaces,
    )
    source("interfaces")

    providers = []
    provider = read("provider")
    for row in provider["results"]:
        configuration = f"{row['shape']}/{'prepared' if row['prepared'] else 'dynamic'}"
        deltas = np.array(
            row["results"]["jaxbind_gpu"]["samples_us"]["throughput"]
        ) - np.array(row["results"]["minimal_ffi"]["samples_us"]["throughput"])
        q1, q3 = np.quantile(deltas, [0.25, 0.75])
        providers.append(
            [
                configuration,
                times(row["results"]["minimal_ffi"]),
                times(row["results"]["jaxbind_gpu"]),
                f"{row['paired_added_us']['latency']:.3f}",
                f"{row['paired_added_us']['throughput']:.3f}",
                f"{q1:.3f} to {q3:.3f}",
            ]
        )
        runtime("provider", configuration, row["results"])
    table(
        "Identical native provider",
        [
            "Configuration",
            "Minimal FFI",
            "Bound C ABI",
            "Paired latency delta",
            "Paired throughput delta",
            "Throughput delta middle 50%",
        ],
        providers,
    )
    lines.extend(
        [
            "Both paths invoke the same cuFINUFFT provider, configuration and plan cache. "
            "Only the native FFI entry point changes. This control isolates bridge cost; "
            "the separate wrapper comparison includes differences between adapters.",
            "",
            f"This control holds {provider['count']} outputs and uses {provider['repeats']} repetitions. "
            "The middle 50% describes repeat-to-repeat variation, not a confidence interval.",
            "",
        ]
    )
    source("provider")

    apps = []
    for name in (
        "integrations",
        "primitives",
        "cached-applications",
        "cached-fft",
        "cached-image",
        "warp",
        "warp-cached",
        "warp-command-buffer",
    ):
        for row in read(name)["applications"]:
            values = row["results"]
            native = values.get("native_return", values["direct_library"])
            apps.append(
                [
                    row["application"]
                    + (" cached" if row["callback_recording_cache"] else "")
                    + (" command buffer" if row["cuda_graphs"] else "")
                    + (
                        f"/{row['workspace_bytes']} B workspace"
                        if row["workspace_bytes"]
                        else ""
                    ),
                    row["size"],
                    times(native),
                    times(values["direct_library"]),
                    times(values["jaxbind_gpu"]),
                    times(values.get("jax_builtin")),
                ]
            )
            runtime(name, f"{row['application']}/{row['size']}", values)
    table(
        "Applications",
        ["Application", "Size", "Native", "Direct adapter", "Bound", "JAX built-in"],
        apps,
    )
    lines.extend(
        [
            "The direct adapter uses the identical algorithms and copies. "
            "The native column uses fresh native results where available. "
            "Equivalent JAX built-ins can use different GPU algorithms. "
            "Sparse diffusion has eight external steps. Torch native "
            "tensors and its stream are prepared once.",
            "",
        ]
    )
    source("integrations")
    source("primitives")
    source("cached-applications")
    source("cached-fft")
    source("cached-image")
    source("warp")
    source("warp-cached")
    source("warp-command-buffer")
    table(
        "Application startup costs",
        ["Run", "Application", "Size", "JAX compilation, ms", "First execution, ms"],
        [
            [
                name,
                row["application"],
                row["size"],
                f"{row['compile_ms']:.2f}",
                f"{row['first_execution_us'] / 1000:.2f}",
            ]
            for name in (
                "integrations",
                "primitives",
                "cached-applications",
                "cached-fft",
                "cached-image",
                "warp",
                "warp-cached",
                "warp-command-buffer",
            )
            for row in read(name)["applications"]
        ],
    )
    lines.extend(
        [
            "First execution includes library setup on that path. External kernel compilation may use an existing disk cache; these measurements do not represent an empty-cache installation.",
            "",
        ]
    )
    table(
        "Warp native graph control",
        ["Mode", "Size", "Native Warp", "Native graph with output copy", "Bound"],
        [
            [
                name,
                row["size"],
                times(row["results"]["native_return"]),
                times(row["results"]["native_cuda_graph_with_output_copy"]),
                times(row["results"]["jaxbind_gpu"]),
            ]
            for name in ("warp", "warp-cached", "warp-command-buffer")
            for row in read(name)["applications"]
        ],
    )
    lines.extend(
        [
            "Warp graph replay copies a stable private output into a fresh result. The binding writes its fresh output directly. Native Warp uses the same sparse product and prepares its input views and stream once.",
            "",
        ]
    )

    graphs = []
    for name in ("graphs", "graphs-additional"):
        for row in read(name)["applications"]:
            values = row["results"]
            graphs.append(
                [
                    row["application"],
                    row["iterations"],
                    times(values["native_cuda_graph_with_output_copy"]),
                    times(values["jaxbind_gpu"]),
                    row["graph_captures_during_timing"],
                ]
            )
            runtime(name, row["application"], values)
    table(
        "Compiled graph applications",
        ["Application", "Steps", "Native graph", "Bound graph", "Python captures"],
        graphs,
    )
    lines.extend(
        [
            "Both programs execute the same 32 external steps. The native "
            "graph copies its stable output to a fresh result, which can "
            "favor the binding. Batches hold eight outputs. Capture counts "
            "include warmup and timing. FFT uses a 24 MiB XLA workspace.",
            "",
        ]
    )

    memory = []
    for name in (
        "memory",
        "graph-memory",
        "graph-memory-additional",
        "cached-memory",
        "cached-image-memory",
        "cached-fft-memory",
        "warp-memory-small",
        "warp-memory-medium",
    ):
        for row in read(name)["results"]:
            stats = row["jax_final_memory_stats"]
            compiler = row["jax_executable_bytes"] or {}
            memory.append(
                [
                    row["application"] + "/" + row["size"],
                    row["variant"],
                    row["count"],
                    row["iterations"],
                    mib(row["output_bytes_per_call"]),
                    f"{mib(row['cupy_peak_active_bytes'])} / {mib(row['cupy_peak_reserved_bytes'])}",
                    f"{mib(stats['peak_bytes_in_use'])} / {mib(stats['peak_pool_bytes'])}",
                    f"{mib(row['torch_peak_active_bytes'])} / {mib(row['torch_peak_reserved_bytes'])}",
                    mib(row.get("warp_sampled_peak_active_bytes")),
                    mib(compiler.get("temp_size_in_bytes")),
                    mib(row["runtime"].get("recording_scratch_bytes", 0)),
                    mib(row["runtime"].get("graph_allocation_bytes", 0)),
                    mib(row.get("native_graph_reserved_bytes")),
                    f"{mib(row['driver_initial_used_bytes'])} / {mib(row['driver_sampled_peak_used_bytes'])}",
                ]
            )
            memory_rows.append(row | {"section": name})
    table(
        "Application memory",
        [
            "Application",
            "Variant",
            "Held outputs",
            "Steps",
            "Output",
            "CuPy live / reserved",
            "JAX live / pool",
            "Torch live / reserved",
            "Warp live peak",
            "XLA scratch",
            "Shared recording scratch",
            "Framework graph scratch",
            "Native graph pool",
            "Driver warm / sampled peak",
        ],
        memory,
    )
    lines.extend(
        [
            "Direct memory rows measure the adapter, including its copies "
            "and conversions. They are not a separate native-return memory "
            "measurement for every application. Library-private scratch "
            "can bypass framework allocators.",
            "",
        ]
    )

    for name in ("dispatch", "ffi-dispatch", "nonlinear"):
        rows, storage = [], []
        for row in read(name)["results"]:
            config = f"{row['size']}" + (f"/{row['mode']}" if "mode" in row else "")
            runtime(name, config, row["results"])
            for variant, timing in row["results"].items():
                rows.append([config, variant, times(timing)])
            for variant, fields in row["jax_executable_bytes"].items():
                storage.append(
                    [
                        config,
                        variant,
                        mib(fields["argument_size_in_bytes"]),
                        mib(fields["output_size_in_bytes"]),
                        mib(fields["temp_size_in_bytes"]),
                    ]
                )
        table(
            name.replace("-", " ") + " runtime",
            ["Elements / mode", "Variant", "Time"],
            rows,
        )
        table(
            name.replace("-", " ") + " compiler storage",
            ["Elements / mode", "Variant", "Arguments", "Outputs", "Scratch"],
            storage,
        )
        source(name)
    table(
        "Python recording cost over native execution",
        [
            "Elements",
            "Cached latency, us",
            "Cached throughput, us",
            "XLA graph latency, us",
            "XLA graph throughput, us",
            "Python captures",
        ],
        [
            [
                r["size"],
                f"{r['paired_python_cached_vs_native_us']['latency']:.3f}",
                f"{r['paired_python_cached_vs_native_us']['throughput']:.3f}",
                f"{r['paired_python_graph_vs_native_graph_us']['latency']:.3f}",
                f"{r['paired_python_graph_vs_native_graph_us']['throughput']:.3f}",
                r["graph_captures_during_timing"],
            ]
            for r in read("dispatch")["results"]
        ],
    )
    intervals = []
    for row in read("dispatch")["results"]:
        samples = row["results"]
        differences = [
            a - b
            for a, b in zip(
                samples["python_cached"]["samples_us"]["throughput"],
                samples["native_binding"]["samples_us"]["throughput"],
            )
        ]
        lower, _, upper = statistics.quantiles(differences, method="inclusive")
        intervals.append([row["size"], f"{lower:.3f} to {upper:.3f}"])
    table(
        "Cached callback paired throughput spread",
        ["Elements", "Added time middle 50%, us"],
        intervals,
    )
    lines.extend(
        [
            "The graph comparison uses identical compiler options and "
            "holds 32 fresh outputs. Zero Python captures does not imply "
            "zero CUDA graph updates. Native and nonlinear derivative "
            "storage above is compiler accounting, not fresh-process "
            "library-private VRAM measurement. Raw files contain paired "
            "samples, cold costs and size-scaling fits.",
            "",
        ]
    )

    paired = []
    for section in ("native_ffi", "derivatives"):
        for row in overview[section]:
            for name, values in row["paired_binding_over_bare_ffi"].items():
                paired.append(
                    [
                        section,
                        row["size"],
                        name,
                        f"{values['latency']['median_us']:.3f}",
                        f"{values['throughput']['median_us']:.3f}",
                        f"{values['throughput']['q1_us']:.3f} to {values['throughput']['q3_us']:.3f}",
                    ]
                )
    table(
        "Added native binding cost over bare FFI",
        [
            "Sweep",
            "Elements",
            "Variant",
            "Latency difference, us",
            "Throughput difference, us",
            "Throughput middle 50%",
        ],
        paired,
    )

    def config(row):
        return (
            f"T{row['kind']}, {'x'.join(map(str, row['grid']))}, "
            f"M={row['points']}, batch={row['batch']}, {row['dtype']}"
        )

    cached = read("cached-wrapper")["results"]
    released = read("released-wrapper")["results"]
    if len(cached) != len(released):
        raise ValueError("Wrapper comparisons must contain the same configurations")
    wrappers = []
    for current, original in zip(cached, released):
        assert config(current) == config(original)
        values = current["timings"]
        wrappers.append(
            [
                config(current),
                times(original["timings"]["jax_finufft"]),
                times(values["wrapper_cached_dynamic"]),
                times(values["jaxbind_gpu_dynamic"]),
                times(values["wrapper_cached_prepared"]),
                times(values["jaxbind_gpu_prepared"]),
            ]
        )
        runtime("cached-wrapper", config(current), values)
        runtime("released-wrapper", config(original), original["timings"])
    table(
        "cuFINUFFT wrappers",
        [
            "Configuration",
            "Released",
            "Cached dynamic",
            "Bound dynamic",
            "Cached prepared",
            "Bound prepared",
        ],
        wrappers,
    )
    lines.extend(
        [
            "The released wrapper creates plans per call. The cached "
            "reference is a benchmark-only build with matching plan and "
            "geometry reuse through its original JAX API. Gains from plan "
            "reuse are not framework gains. Released and cached timings "
            "come from separate runs. Precision, signs and library options "
            "match; the raw files include every option and cold timing.",
            "",
        ]
    )
    variants = (
        "released_dynamic",
        "cached_dynamic",
        "bound_dynamic",
        "cached_prepared",
        "bound_prepared",
    )
    wrapper_memory = read("wrapper-memory")["results"]
    for title, field in (
        ("Wrapper sampled driver peaks", "driver_sampled_peak_used_bytes"),
        ("Wrapper JAX live peaks", "jax_live"),
    ):
        rows = []
        for case in cached:
            expected = {key: case[key] for key in wrapper_memory[0]["case"]}
            values = {r["variant"]: r for r in wrapper_memory if r["case"] == expected}
            rows.append(
                [config(case)]
                + [
                    mib(
                        values[v]["jax_final_memory_stats"]["peak_bytes_in_use"]
                        if field == "jax_live"
                        else values[v][field]
                    )
                    for v in variants
                ]
            )
        table(title, ["Configuration", *variants], rows)
    memory_rows.extend(
        row | {"section": "wrapper-memory", "configuration": config(row["case"])}
        for row in wrapper_memory
    )
    source("wrapper-memory")

    allocation_memory = read("output-allocation-memory")["results"]
    rows = []
    for row in read("output-allocation")["results"]:
        values = row["timings"]
        rows.append(
            [
                row["app"],
                row["size"],
                *(times(values[v]) for v in ("native", "copy", "allocate")),
                f"{row['paired_saved_us']['throughput']:.2f}",
                all(row["result_pointer_is_xla_output"]["allocate"]),
            ]
        )
        runtime("output-allocation", f"{row['app']}/{row['size']}", values)
    table(
        "Return-only allocation policies",
        [
            "Function",
            "Elements",
            "Native",
            "Copy",
            "Forward",
            "Paired saved throughput, us",
            "Forwarded pointer",
        ],
        rows,
    )
    table(
        "Allocation policy memory",
        ["Function", "Variant", "Extra CuPy live", "JAX live / pool", "XLA scratch"],
        [
            [
                r["app"],
                r["variant"],
                mib(r["cupy_peak_additional_active_bytes"]),
                f"{mib((r['jax_final_memory_stats'] or {}).get('peak_bytes_in_use'))} / {mib((r['jax_final_memory_stats'] or {}).get('peak_pool_bytes'))}",
                mib((r["executable_bytes"] or {}).get("temp_size_in_bytes")),
            ]
            for r in allocation_memory
        ],
    )
    memory_rows.extend(
        row | {"section": "output-allocation-memory"} for row in allocation_memory
    )
    source("output-allocation")
    source("output-allocation-memory")
    lines.extend(
        [
            "Full tables are available as [runtime CSV](current-runtime.csv) and "
            "[memory CSV](current-memory-comparison.csv). Memory CSV includes "
            "every counter, with nested field names separated by dots. Raw "
            "results retain all samples and source hashes.",
            "",
        ]
    )
    (directory / "current-comparison.md").write_text("\n".join(lines))

    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            else:
                result[name] = json.dumps(item) if isinstance(item, list) else item
        return result

    for name, rows in (("runtime", timing_rows), ("memory-comparison", memory_rows)):
        rows = [flatten(row) for row in rows]
        fields = sorted({key for row in rows for key in row})
        with (directory / f"current-{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
