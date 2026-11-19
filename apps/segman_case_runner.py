"""New-case directory adapter; original model, weights and inference remain unchanged.

The native v1 wrapper omits its relocated artifacts directory. This additive
adapter records its identity without rewriting that frozen wrapper or registry.
"""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path


def initialize(job):
    job.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "artifacts"):
        (job / name).mkdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "original_component", root / "Algorithm_Tool/segman_slo/runner.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    image, output = args.image.resolve(strict=True), args.observation.resolve()
    job = output.parent.parent
    if not image.is_relative_to(root / ".local/13-new-slo-v1") or not job.is_relative_to(
        root / ".local/13-new-slo-v1"
    ):
        raise ValueError("new_case_runner_path")
    if output.name != "result.json" or output.parent.name != "prediction" or job.exists():
        raise ValueError("new_case_runner_new_output_required")
    lock = json.loads((root / "configs/tools/segman-slo-resources-v1.json").read_bytes())
    for entry in lock["resources"]:
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or original.sha(path) != entry["sha256"]:
            raise ValueError("segman_resource_drift")
    with (root / ".local/segman-component.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(job)
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if device != "0":
            raise ValueError("new_case_single_device_required")
        os.environ.clear()
        os.environ.update(
            PATH="/opt/data/test/envs/SegMAN-SLO/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            CUDA_VISIBLE_DEVICES=device,
            PYTHONNOUSERSITE="1",
            TMPDIR=str(job / "tmp"),
            XDG_CACHE_HOME=str(job / "cache"),
            MPLCONFIGDIR=str(job / "cache/matplotlib"),
            TRITON_CACHE_DIR=str(job / "cache/triton"),
            CUDA_CACHE_PATH=str(job / "cache/cuda"),
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        spec = importlib.util.spec_from_file_location(
            "fixed_segman", original.DEPLOYMENT / "scripts/segman_slo.py"
        )
        model_owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_owner)
        model_owner.ROOT = job
        model = model_owner.build_strict_model()
        result = model_owner.infer_image(model, image, output.parent, "cuda:0")
        print(json.dumps(dict(status=result["status"], observation=str(output))))


if __name__ == "__main__":
    main()
