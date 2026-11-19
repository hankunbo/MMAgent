"""Same installed SegMAN inference; only the versioned case workspace is new."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True


def resource_owner():
    # This runner uses the installed model's isolated Python, without project packages.
    spec = importlib.util.spec_from_file_location(
        "segman_resource_owner", Path(__file__).with_name("segman_resource_lock.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def execution_resource(root, image):
    binding = json.loads((image.parent / "binding.json").read_bytes())
    if binding["schema_version"] == "generic-case-binding/1.2.0":
        owner = resource_owner()
        owner.verify_input(root, image.parent, binding)
        with owner.resource_lock(root):
            yield
    else:
        with (root / ".local/segman-component.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def checked_paths(root, image, output):
    image, output = image.resolve(strict=True), output.resolve()
    job, case = output.parent.parent, image.parent
    if (
        case.parent != (root / ".local/fairvlmed-runs").resolve()
        or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", case.name)
        or image.name != "input.png"
        or job.parent != case
        or job.name != "segman-r03"
        or output.name != "result.json"
        or output.parent.name != "prediction"
    ):
        raise ValueError("generic_segman_workspace_binding")
    binding = json.loads((case / "binding.json").read_bytes())
    if (
        # 1.1 adds separately authorized age context. Inference still consumes
        # only this exact image; the namespace, bytes and output checks remain.
        binding["schema_version"]
        not in {
            "generic-case-binding/1.0.0",
            "generic-case-binding/1.1.0",
            "generic-case-binding/1.2.0",
        }
        or binding["execution_namespace"] != case.name
        or "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()
        != binding["image"]["checksum"]
    ):
        raise ValueError("generic_segman_image_binding")
    if binding["schema_version"] == "generic-case-binding/1.2.0":
        resource_owner().verify_input(root, case, binding)
    if job.exists():
        raise ValueError("generic_segman_new_output_required")
    return image, output, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image, output, job = checked_paths(root, args.image, args.observation)
    spec = importlib.util.spec_from_file_location(
        "original_component", root / "Algorithm_Tool/segman_slo/runner.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    resources = json.loads((root / "configs/tools/segman-slo-resources-v1.json").read_bytes())
    for entry in resources["resources"]:
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or original.sha(path) != entry["sha256"]:
            raise ValueError("segman_resource_drift")
    with execution_resource(root, image):
        job.mkdir(parents=True, exist_ok=False)
        for name in ("tmp", "artifacts"):
            (job / name).mkdir()
        if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
            raise ValueError("generic_segman_single_device_required")
        os.environ.clear()
        os.environ.update(
            PATH="/opt/data/test/envs/SegMAN-SLO/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            CUDA_VISIBLE_DEVICES="0",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
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
