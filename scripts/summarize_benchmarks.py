"""Build the current metric overview from versioned benchmark evidence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import jaxbind_gpu as jb


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"


def load(name):
    path = RESULTS / f"current-{name}.json"
    data = json.loads(path.read_text())
    if "execution_package" in data:
        versions = [data["execution_package"]["version"]]
    elif "package_version" in data:
        versions = [data["package_version"]]
    else:
        versions = [row["package_version"] for row in data["results"]]
    if not versions or any(version != jb.__version__ for version in versions):
        raise ValueError(f"{path.name} must be measured with version {jb.__version__}")
    package = Path(jb.__file__).resolve().parent
    hashes = [
        data.get("execution_package", {}).get("sha256", {}),
        data.get("source_sha256", {}),
        data.get("package_sha256", {}),
    ]
    hashes.extend(row.get("source_sha256", {}) for row in data.get("results", []))
    for recorded in hashes:
        for filename, expected in recorded.items():
            candidate = ROOT / filename
            if not candidate.is_file():
                candidate = package / filename
            if not candidate.is_file():
                raise ValueError(f"{path.name} is missing measured source {filename}")
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{path.name} was measured with different {filename}")
    jobs = json.loads((ROOT / "benchmarks/suite.json").read_text())
    script = next(ROOT / job["arguments"][0] for job in jobs if job["name"] == name)
    for field, candidate in (
        ("script_sha256", script),
        ("benchmark_support_sha256", ROOT / "scripts/benchmark_common.py"),
    ):
        if (
            field in data
            and hashlib.sha256(candidate.read_bytes()).hexdigest() != data[field]
        ):
            raise ValueError(
                f"{path.name} was measured with different {candidate.name}"
            )
    return path, data


def spread(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def native_rows(data, evidence):
    rows = []
    for row in data["results"]:
        paired = {}
        for name in row["paired_native_vs_bare_ffi_us"]:
            suffix = (
                "_jvp"
                if name.endswith("_jvp")
                else "_vjp"
                if name.endswith("_vjp")
                else ""
            )
            paired[name] = {}
            for metric in ("latency", "throughput"):
                difference = np.array(
                    row["results"][name]["samples_us"][metric]
                ) - np.array(row["results"]["bare_ffi" + suffix]["samples_us"][metric])
                paired[name][metric] = {
                    "median_us": float(np.median(difference)),
                    "q1_us": float(np.quantile(difference, 0.25)),
                    "q3_us": float(np.quantile(difference, 0.75)),
                }
        rows.append(
            {
                "size": row["size"],
                "array_bytes": row["array_bytes"],
                "paired_binding_over_bare_ffi": paired,
                "hlo_custom_calls": row["hlo_custom_calls"],
                "executable_memory": row["jax_executable_bytes"],
                "evidence": evidence,
            }
        )
    return rows


def main():
    suite = json.loads((ROOT / "benchmarks/suite.json").read_text())
    commands = json.loads((RESULTS / "current-commands.json").read_text())
    completed = {job["name"] for job in commands["jobs"] if job["exit_code"] == 0}
    missing = {job["name"] for job in suite} - completed
    if commands["package_version"] != jb.__version__ or missing:
        raise ValueError(
            f"Complete the current benchmark suite before reporting: {sorted(missing)}"
        )
    overview = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": jb.__version__,
        "devices": {},
        "units": {"timings": "microseconds", "memory": "bytes"},
        "methodology": "Warmed medians; changing resident inputs; fresh held outputs; independent correctness checks; immediate metric warm batches; alternating variant order; exclusive GPU",
        "baseline_contracts": {
            "direct_library": "Identical adapter algorithms, conversions, fresh output allocations and result copies",
            "native_return": "Fresh library result without adapter copy; native Torch/Warp views and streams prepared once",
            "native_cuda_graph_with_output_copy": "Identical captured external steps with stable graph output copied into a fresh retained result",
            "bare_ffi": "Same externally compiled CUDA scale kernel through a minimal typed JAX FFI handler",
            "allocation_forwarding": "Same return-only function and persistent-resource strategy; only output allocation policy changes",
        },
        "applications": [],
        "graphs": [],
        "memory": [],
        "native_ffi": [],
        "derivatives": [],
        "wrapper_comparisons": {},
        "evidence_sha256": {},
    }

    def read(name):
        path, data = load(name)
        evidence = str(path.relative_to(ROOT))
        overview["evidence_sha256"][evidence] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for gpu in [
            data.get("gpu", {}),
            *[row.get("gpu", {}) for row in data.get("results", [])],
        ]:
            if "uuid" in gpu:
                overview["devices"][gpu["uuid"]] = gpu
        return evidence, data

    evidence, interfaces = read("interfaces")
    overview["interfaces"] = {
        "applications": interfaces["applications"],
        "evidence": evidence,
    }
    evidence, provider = read("provider")
    overview["provider_control"] = {
        "results": provider["results"],
        "methodology": provider["methodology"],
        "evidence": evidence,
    }

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
        evidence, data = read(name)
        for row in data["applications"]:
            results = row["results"]
            baseline = (
                "native_return" if "native_return" in results else "direct_library"
            )
            metrics = {}
            for metric in ("latency", "throughput"):
                values = {
                    name: result[metric + "_us"] for name, result in results.items()
                }
                metrics[metric] = {
                    "variant_us": values,
                    "native_baseline": baseline,
                    "binding_over_adapter_us": values["jaxbind_gpu"]
                    - values["direct_library"],
                    "adapter_over_native_us": values["direct_library"]
                    - values[baseline],
                    "binding_over_native_percent": 100
                    * (values["jaxbind_gpu"] / values[baseline] - 1),
                }
            overview["applications"].append(
                {
                    key: row[key]
                    for key in (
                        "application",
                        "size",
                        "backend",
                        "steps",
                        "inputs",
                        "callback_recording_cache",
                        "cuda_graphs",
                        "workspace_bytes",
                    )
                }
                | {"metrics": metrics, "evidence": evidence}
            )
    for name in ("graphs", "graphs-additional"):
        evidence, data = read(name)
        for row in data["applications"]:
            metrics = {}
            for metric in ("latency", "throughput"):
                values = {
                    name: result[metric + "_us"]
                    for name, result in row["results"].items()
                }
                metrics[metric] = {
                    "variant_us": values,
                    "bound_over_native_graph_percent": 100
                    * (
                        values["jaxbind_gpu"]
                        / values["native_cuda_graph_with_output_copy"]
                        - 1
                    ),
                }
            overview["graphs"].append(
                {
                    "application": row["application"],
                    "iterations": row["iterations"],
                    "workspace_bytes": row["workspace_bytes"],
                    "capture_count": row["graph_captures_during_timing"],
                    "capture_scope": row["graph_capture_scope"],
                    "metrics": metrics,
                    "evidence": evidence,
                }
            )
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
        evidence, data = read(name)
        overview["memory"].extend(
            row | {"evidence": evidence} for row in data["results"]
        )
    for name, key in (("dispatch", "native_ffi"), ("ffi-dispatch", "derivatives")):
        evidence, data = read(name)
        overview[key] = native_rows(data, evidence)
        if name == "dispatch":
            overview["constant_cost_claim"] = {
                "supported": "Native bridge CPU bookkeeping is independent of array length for fixed buffer count and rank; no GPU staging allocation by the bridge",
                "scope": "Wall-clock differences are measurements for the reported configurations; no universal overhead bound is inferred",
                "size_scaling_fits": data["size_scaling"],
            }
            overview["graph_dispatch"] = [
                {
                    key: row[key]
                    for key in (
                        "size",
                        "array_bytes",
                        "graph_captures_during_timing",
                        "paired_graph_binding_vs_bare_ffi_us",
                        "paired_python_graph_vs_native_graph_us",
                        "paired_python_cached_vs_native_us",
                    )
                }
                | {"evidence": evidence}
                for row in data["results"]
            ]
    evidence, data = read("nonlinear")
    overview["nonlinear_derivatives"] = [
        row | {"evidence": evidence} for row in data["results"]
    ]
    for name in ("cached-wrapper", "released-wrapper"):
        evidence, data = read(name)
        metrics = {}
        for suffix in ("dynamic", "prepared"):
            metrics[suffix] = {}
            for metric in ("latency", "throughput"):
                values = []
                for row in data["results"]:
                    timings = row["timings"]
                    ours = timings["jaxbind_gpu_" + suffix][metric + "_us"]
                    reference = timings[
                        "wrapper_cached_" + suffix
                        if name == "cached-wrapper"
                        else "jax_finufft"
                    ][metric + "_us"]
                    values.append(
                        ours / reference
                        if name == "cached-wrapper"
                        else reference / ours
                    )
                metrics[suffix][
                    metric + ("_ratio" if name == "cached-wrapper" else "_speedup")
                ] = spread(values)
        overview["wrapper_comparisons"]["current-" + name] = {
            "configurations": len(data["results"]),
            "reference_build": data["reference_build"],
            "metrics": metrics,
            "evidence": evidence,
        }
    evidence, data = read("wrapper-memory")
    overview["wrapper_memory"] = [
        row | {"evidence": evidence} for row in data["results"]
    ]
    for name, key in (
        ("output-allocation", "output_allocation"),
        ("output-allocation-memory", "output_allocation_memory"),
    ):
        evidence, data = read(name)
        overview[key] = [row | {"evidence": evidence} for row in data["results"]]
    overview["graph_exclusions"] = [
        {
            "application": "sparse_diffusion",
            "reason": "CuPy path attempts D2H while capturing",
            "evidence": "benchmarks/results/logs/current-sparse-graphs.log",
        },
        {
            "application": "batched_gemm",
            "reason": "CuPy cuBLAS path rejects stream capture",
            "evidence": "benchmarks/results/logs/current-gemm-graph-check.log",
        },
    ]
    overview["hardware"] = ", ".join(
        sorted({gpu["name"] for gpu in overview["devices"].values()})
    )
    overview["devices"] = list(overview["devices"].values())
    (RESULTS / "current-overview.json").write_text(
        json.dumps(overview, indent=2) + "\n"
    )
    from report_benchmarks import write_report

    write_report(overview, RESULTS)
    print(f"Summarized {len(overview['evidence_sha256'])} current evidence files")


if __name__ == "__main__":
    main()
