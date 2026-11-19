"""CPU-only mask checks and reuse of existing geometric measurement algorithms."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/opt/data/test/MMAgent")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect(directory, image):
    report = json.loads((directory / "result.json").read_bytes())
    image_bytes = image.read_bytes()
    if report["status"] != "success" or hashlib.sha256(image_bytes).hexdigest() != report["input_sha256"]:
        raise ValueError("segman_success_input_binding")
    shape = np.asarray(Image.open(image)).shape
    labels = np.asarray(Image.open(directory / "labels.png"))
    if labels.dtype != np.uint8 or labels.shape != shape or not set(np.unique(labels)) <= {0, 1, 2}:
        raise ValueError("segman_label_shape_or_class")
    masks = {}
    for name, expected in (("cup", labels == 2), ("rim", labels == 1), ("disc", labels > 0)):
        raw = np.asarray(Image.open(directory / (name + ".png")))
        if raw.dtype != np.uint8 or raw.shape != shape or not np.array_equal(raw, expected.astype(np.uint8) * 255):
            raise ValueError("segman_binary_union_mismatch")
        if int(expected.sum()) != report["pixel_counts"][name] or not expected.any():
            raise ValueError("segman_pixel_count_mismatch")
        masks[name] = expected
    for name, ref in report["outputs"].items():
        if Path(name).name != name:
            raise ValueError("segman_output_path")
        body = (directory / name).read_bytes()
        if len(body) != ref["bytes"] or hashlib.sha256(body).hexdigest() != ref["sha256"]:
            raise ValueError("segman_output_hash")
    cdr_path = ROOT / "Algorithm_Tool/cdr_metrics/runner.py"
    geometry_path = ROOT / "Algorithm_Tool/mask_metrics/runner.py"
    cdr = load("existing_cdr", cdr_path)
    geometry = load("existing_geometry", geometry_path)
    results = {m: cdr.calculate(m, directory / "disc.png", directory / "cup.png") for m in ("vcdr", "acdr")}
    results.update(hcdr=geometry.calculate_hcdr(masks["disc"], masks["cup"]), rdar=geometry.calculate_rdar(masks["disc"], masks["cup"]), rim_profile=geometry.calculate_rim_profile(masks["disc"], masks["cup"]))
    return {
        "schema": "segman-case-mask-check/1.0.0", "shape_hw": list(shape),
        "input_sha256": report["input_sha256"], "pixel_counts": report["pixel_counts"],
        "metrics": results, "mask_relationships_verified": True,
        "algorithm_sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (cdr_path, geometry_path)],
        "geometry_scope": "image_axes_pixels_and_ratios; no_mm_scale_or_anatomical_quadrant_inference",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(inspect(a.directory, a.image), ensure_ascii=False, allow_nan=False))
