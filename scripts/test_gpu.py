"""Run the small CUDA test suite on one currently idle GPU."""

import os
import subprocess
import sys


def idle_devices(count=1):
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    devices = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None if visibility is None else visibility.split(",")
    selected = []
    for line in devices:
        index, uuid, memory, utilization = [item.strip() for item in line.split(",")]
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if uuid not in busy and int(memory) < 128 and int(utilization) == 0:
            selected.append(uuid)
            if len(selected) == count:
                return selected
    raise SystemExit(f"Fewer than {count} idle GPUs within CUDA_VISIBLE_DEVICES")


def idle_device():
    return idle_devices()[0]


if __name__ == "__main__":
    device = idle_device()
    print(f"Testing on idle device {device}", flush=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "pytest", *sys.argv[1:]], env=env)
    )
