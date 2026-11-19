"""Fixed component adapter. Keeps per-run audits out of original deployment evidence."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path("/opt/data/test/MMAgent")
DEPLOYMENT = Path("/opt/data/test/segman-slo-deployment")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--observation", type=Path, required=True)
    a = p.parse_args()
    image = a.image.resolve(strict=True)
    out = a.observation.resolve()
    job = out.parent.parent
    if not image.is_relative_to("/opt/data/test") or not job.is_relative_to(ROOT / ".local"):
        raise ValueError("segman_component_path_scope")
    if out.name != "result.json" or out.parent.name != "prediction" or job.exists():
        raise ValueError("segman_new_job_prediction_result_required")
    spec = json.loads((ROOT / "configs/tools/segman-slo-resources-v1.json").read_bytes())
    for r in spec["resources"]:
        path = Path(r["path"])
        if path.stat().st_size != r["bytes"] or sha(path) != r["sha256"]:
            raise ValueError("segman_resource_drift")
    with (ROOT / ".local/segman-component.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job.mkdir(parents=True, exist_ok=False)
        (job / "tmp").mkdir()
        # No inherited database/API credentials. Visibility is selection, not isolation.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if "," in visible or not visible:
            raise ValueError("segman_single_device_required")
        os.environ.clear()
        os.environ.update(PATH="/opt/data/test/envs/SegMAN-SLO/bin:/usr/bin:/bin", CUDA_VISIBLE_DEVICES=visible, PYTHONNOUSERSITE="1", TMPDIR=str(job / "tmp"), XDG_CACHE_HOME=str(job / "cache"), MPLCONFIGDIR=str(job / "cache/matplotlib"), TRITON_CACHE_DIR=str(job / "cache/triton"), CUDA_CACHE_PATH=str(job / "cache/cuda"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        module_spec = importlib.util.spec_from_file_location("fixed_segman", DEPLOYMENT / "scripts/segman_slo.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.ROOT = job  # only audit destination; fixed SOURCE/WEIGHT/CONFIG remain intact
        model = module.build_strict_model()
        result = module.infer_image(model, image, out.parent, "cuda:0")
        print(json.dumps({"status": result["status"], "observation": str(out)}))


if __name__ == "__main__":
    main()
