"""Run the benchmark suite sequentially per GPU and record commands and logs.

Supply one GPU UUID for a serial run, or two for separate runtime and memory
queues. Each job independently checks that its assigned device is idle.
Optional libraries and reference builds are required by the full suite.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--only", nargs="+", help="Select named jobs")
    parser.add_argument("--wait", type=float, default=60)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.devices and (
        len(args.devices) > 2 or len(set(args.devices)) != len(args.devices)
    ):
        parser.error("Supply one or two distinct GPU UUIDs")
    jobs = json.loads((ROOT / "benchmarks/suite.json").read_text())
    if args.only:
        unknown = set(args.only) - {job["name"] for job in jobs}
        if unknown:
            parser.error(f"Unknown jobs: {sorted(unknown)}")
        jobs = [job for job in jobs if job["name"] in args.only]
    if args.list:
        for job in jobs:
            print(job["name"], *job["arguments"])
        return
    if not args.devices:
        parser.error("Supply --devices with one or two idle GPU UUIDs")
    results = ROOT / "benchmarks/results"
    (results / "logs").mkdir(parents=True, exist_ok=True)
    commands = results / "current-commands.json"
    version = importlib.metadata.version("jaxbind-gpu")
    payload = {
        "package_version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    if args.only and commands.exists():
        payload = json.loads(commands.read_text())
        if payload["package_version"] != version:
            raise ValueError("Cannot combine command records from different versions")
        selected = {job["name"] for job in jobs}
        payload["jobs"] = [
            job for job in payload["jobs"] if job["name"] not in selected
        ]
    condition = threading.Condition()
    failed = threading.Event()
    completed = {row["name"] for row in payload["jobs"] if row["exit_code"] == 0}
    selected = {job["name"] for job in jobs}
    for job in jobs:
        missing = set(job.get("requires", [])) - completed - selected
        if missing:
            parser.error(f"{job['name']} requires completed jobs: {sorted(missing)}")

    def queue(device, assigned):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=device,
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        for job in assigned:
            with condition:
                condition.wait_for(
                    lambda: failed.is_set()
                    or all(name in completed for name in job.get("requires", []))
                )
            if failed.is_set():
                break
            command = [
                sys.executable,
                str(ROOT / "scripts/run_gpu.py"),
                "--wait",
                str(args.wait),
                *job["arguments"],
                "--output",
                str(results / f"current-{job['name']}.json"),
            ]
            print(f"Starting {job['name']} on {device}", flush=True)
            start = time.monotonic()
            with (results / "logs" / f"current-{job['name']}.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            expected = job.get("expected_exit", 0)
            passed = result.returncode == expected
            if job.get("expected_error"):
                passed = (
                    passed
                    and job["expected_error"]
                    in (results / "logs" / f"current-{job['name']}.log").read_text()
                )
            exit_code = 0 if passed else 1
            row = {
                "name": job["name"],
                "command": command,
                "exit_code": exit_code,
                "actual_exit_code": result.returncode,
                "expected_exit_code": expected,
                "seconds": time.monotonic() - start,
                "gpu_uuid": device,
            }
            with condition:
                payload["jobs"].append(row)
                commands.write_text(json.dumps(payload, indent=2) + "\n")
                if exit_code:
                    failed.set()
                else:
                    completed.add(job["name"])
                condition.notify_all()
            print(
                f"Finished {job['name']}: check {'passed' if passed else 'failed'}, {row['seconds']:.1f} s",
                flush=True,
            )
            if exit_code:
                break

    if len(args.devices) == 1:
        queue(args.devices[0], jobs)
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    queue, device, [job for job in jobs if job["kind"] == kind]
                )
                for device, kind in zip(args.devices, ("runtime", "memory"))
            ]
            for future in futures:
                future.result()
    if failed.is_set():
        raise SystemExit(
            "Benchmark job failed; inspect its log before resuming with --only"
        )


if __name__ == "__main__":
    main()
