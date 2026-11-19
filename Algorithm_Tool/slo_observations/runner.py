"""Fixed CPU SLOctolyzer adapters and four approved, non-diagnostic measurements.

Uses the separately installed SLOctolyzer environment; never imports it into ENV-001.
The registered output JSON is a component observation, not a Runtime/clinical verdict.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

ROOT = Path("/opt/data/test/MMAgent")
SOURCE = Path("/opt/data/test/SLOctolyzer")
LOCK = ROOT / "configs/tools/sloctolyzer-resources-v1.json"
METRICS = ("density", "width", "tortuosity", "fractal")
MODELS = ("vessels", "avod", "fovea")
MAX_PIXELS = 4_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_installation():
    lock = json.loads(LOCK.read_text())
    for item in lock["resources"]:
        p = Path(item["path"])
        if p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise ValueError("installed_resource_drift")
    return sha(LOCK)


def read_image(path, *, mask=False):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to("/opt/data/test") or path.stat().st_size > 32_000_000:
        raise ValueError("input_path_or_size")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "PNG" or im.mode != "L" or im.width * im.height > MAX_PIXELS:
            raise ValueError("expected_bounded_grayscale_png")
        arr = np.asarray(im).copy()
    if mask and not set(np.unique(arr)).issubset({0, 255}):
        raise ValueError("expected_binary_0_255_mask")
    return arr, {
        "uri": path.as_uri(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "shape_hw": list(arr.shape),
    }


def segments(skeleton):
    """Deterministic 8-neighbour paths; drop diagonal shortcut if axial connection exists.

    Junctions delimit paths. Every edge is visited once. Pure loops are reported but
    have no arc/chord value. This is our explicit path extraction, not Retipy density.
    """
    points = set(map(tuple, np.argwhere(skeleton)))
    if len(points) > 250_000:
        raise ValueError("skeleton_budget_exceeded")

    def adjacent(p):
        r, c = p
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if (dr or dc) and q in points:
                    if dr and dc and ((r + dr, c) in points or (r, c + dc) in points):
                        continue
                    out.append(q)
        return sorted(out)

    neighbours = {p: adjacent(p) for p in sorted(points)}
    used = set()
    paths = []
    for start in sorted(points, key=lambda p: (len(neighbours[p]) == 2, p)):
        for nxt in neighbours[start]:
            edge = tuple(sorted((start, nxt)))
            if edge in used:
                continue
            used.add(edge)
            path = [start, nxt]
            previous, current = start, nxt
            while current != start and len(neighbours[current]) == 2:
                q = next(p for p in neighbours[current] if p != previous)
                edge = tuple(sorted((current, q)))
                if edge in used:
                    break
                used.add(edge)
                path.append(q)
                previous, current = current, q
            paths.append(path)
    return paths


def measure(mask, task):
    if task not in METRICS or mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("measurement_input")
    result = {
        "metric": task,
        "value": None,
        "unavailable_reason": None,
        "unit": "pixel" if task == "width" else "dimensionless",
        "definition_version": "slo-measurement/1.0.0",
        "roi": "entire_image_including_background",
        "shape_hw": list(mask.shape),
    }
    count = int(mask.sum())
    if not count:
        result["unavailable_reason"] = "empty_vessel_mask_not_a_negative_diagnosis"
        return result
    if task == "density":
        result["value"] = count / mask.size
        result["numerator"], result["denominator"] = count, int(mask.size)
    elif task == "width":
        n = int(skeletonize(mask, method="zhang").sum())
        result["value"] = count / n if n else None
        result["unavailable_reason"] = None if n else "empty_skeleton"
        result["vessel_pixels"], result["skeleton_pixels"] = count, n
        result["physical_scale"] = None
    elif task == "fractal":
        # Exact non-empty/non-full box counting used in the published toolkit.
        n = int(np.floor(np.log2(min(mask.shape))))
        sizes = (2 ** np.arange(n, 1, -1)).tolist()
        counts = []
        for k in sizes:
            sums = np.add.reduceat(
                np.add.reduceat(mask.astype(int), np.arange(0, mask.shape[0], k), axis=0),
                np.arange(0, mask.shape[1], k),
                axis=1,
            )
            counts.append(int(((sums > 0) & (sums < k * k)).sum()))
        result["box_sizes"], result["box_counts"] = sizes, counts
        if len(sizes) < 3 or any(c == 0 for c in counts):
            result["unavailable_reason"] = "insufficient_valid_box_scales"
        else:
            result["value"] = float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])
    else:
        accepted, rejected = [], {"closed_or_zero_chord": 0, "shorter_than_10px": 0}
        for path in segments(skeletonize(mask, method="zhang")):
            p = np.asarray(path, dtype=float)
            length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            chord = float(np.linalg.norm(p[-1] - p[0]))
            if chord == 0:
                rejected["closed_or_zero_chord"] += 1
            elif length < 10:
                rejected["shorter_than_10px"] += 1
            else:
                accepted.append(
                    {
                        "start_yx": p[0].astype(int).tolist(),
                        "end_yx": p[-1].astype(int).tolist(),
                        "length_px": length,
                        "chord_px": chord,
                        "arc_chord_ratio": length / chord,
                    }
                )
        result["segments"], result["excluded"] = accepted, rejected
        result["aggregation"] = "unweighted_median_of_valid_segments"
        result["path_policy"] = "zhang-8neighbour-no-diagonal-shortcut/min10px/1.0.0"
        if accepted:
            result["value"] = float(np.median([s["arc_chord_ratio"] for s in accepted]))
        else:
            result["unavailable_reason"] = "no_valid_open_segments"
    if result["value"] is not None and not math.isfinite(result["value"]):
        raise ValueError("nonfinite_measurement")
    return result


def predict(arr, task):
    if task not in MODELS or arr.max() == arr.min():
        raise ValueError("model_task_or_constant_image")
    # CPU is the observed deployment. Never download a checkpoint or use another device.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(SOURCE))
    import torch
    from sloctolyzer.segment import avo_inference, fov_inference, slo_inference

    torch.set_num_threads(8)
    torch.manual_seed(0)
    classes = {
        "vessels": (slo_inference.SLOSegmenter, "slosegmenter"),
        "avod": (avo_inference.AVOSegmenter, "avosegmenter"),
        "fovea": (fov_inference.FOVSegmenter, "fovsegmenter"),
    }
    cls, name = classes[task]
    model = cls(local_model_path=SOURCE / "weights" / (name + "_weights.pth"))
    # Array entry reproduces analyse.py's per-image intensity rescaling branch.
    pred = model.predict_img(arr)
    fields = {
        "device": model.device,
        "torch": torch.__version__,
        "preprocessing": "official_array_branch_native_shape_restore",
        "eye": "unknown",
        "physical_scale": None,
    }
    if task == "vessels":
        # analyse.py:354: fuse binary and AV detections, remove the optic disc.
        # The public vessel tool therefore explicitly depends on BOTH trained models.
        avo = avo_inference.AVOSegmenter(
            local_model_path=SOURCE / "weights/avosegmenter_weights.pth"
        )
        avod, _ = avo.predict_img(arr, location=None)
        fused = (
            (pred + (avod[..., [0, 2]].sum(axis=-1) > 0)).astype(bool) * (1 - avod[..., 1])
        ).astype(int)
        arrays = {"vessel_mask": fused}
        fields["postprocessing"] = "official_analyse_binary_union_AV_minus_disc"
        fields["model_dependencies"] = ["slosegmenter", "avosegmenter"]
    elif task == "avod":
        avod, centre = pred
        arrays = {"artery_mask": avod[..., 0], "vein_mask": avod[..., 2], "disc_mask": avod[..., 1]}
        fields["disc_centre_xy"] = None if centre is None else np.asarray(centre).tolist()
    else:
        heatmap, point = pred
        if (
            heatmap.shape != arr.shape
            or not np.isfinite(heatmap).all()
            or heatmap.min() < 0
            or heatmap.max() > 1
        ):
            raise ValueError("invalid_fovea_heatmap")
        x, y = map(float, point)
        if not (0 <= x < arr.shape[1] and 0 <= y < arr.shape[0]):
            raise ValueError("invalid_fovea_coordinate")
        fields["fovea_xy"] = [x, y]
        fields["fovea_status"] = "unavailable_origin_sentinel" if x == y == 0 else "predicted"
        arrays = {"fovea_map": heatmap.astype(np.float32)}
    for key, value in arrays.items():
        if key.endswith("_mask") and (
            value.shape != arr.shape
            or not np.isfinite(value).all()
            or not np.isin(value, [0, 1]).all()
        ):
            raise ValueError("invalid_segmentation_output")
    fields["foreground_pixels"] = {
        k: int(v.sum()) for k, v in arrays.items() if k.endswith("_mask")
    }
    return arrays, fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODELS + METRICS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="observation JSON")
    for key in ("vessel_mask", "artery_mask", "vein_mask", "disc_mask", "fovea_map"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    args = parser.parse_args(argv)
    required = {
        "vessels": ["vessel_mask"],
        "avod": ["artery_mask", "vein_mask", "disc_mask"],
        "fovea": ["fovea_map"],
    }.get(args.task, [])
    given = [
        k
        for k in ("vessel_mask", "artery_mask", "vein_mask", "disc_mask", "fovea_map")
        if getattr(args, k) is not None
    ]
    if set(given) != set(required):
        raise ValueError("output_ports_mismatch")
    destinations = [args.output] + [getattr(args, k) for k in required]
    if len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError("duplicate_output_path")
    for p in destinations:
        if p.exists() or not p.resolve().is_relative_to("/opt/data/test"):
            raise ValueError("output_exists_or_outside_root")
    lock_sha = verify_installation()
    arr, source = read_image(args.input, mask=args.task in METRICS)
    output = {
        "schema_version": "slo-observation/1.0.0",
        "task": args.task,
        "research_only": True,
        "clinical_effect": None,
        "input": source,
        "installation_sha256": lock_sha,
        "correlation_group": source["sha256"],
        "outputs": {},
    }
    if args.task in METRICS:
        arrays = {}
        output["measurement"] = measure(arr > 0, args.task)
    else:
        arrays, output["model"] = predict(arr, args.task)
    # Reject all bad values before any output, JSON is the completion marker.
    json.dumps(output, allow_nan=False)
    created = []
    try:
        for key, value in arrays.items():
            p = getattr(args, key)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as f:
                created.append(p)
                if key == "fovea_map":
                    np.save(f, value, allow_pickle=False)
                else:
                    Image.fromarray((value * 255).astype(np.uint8)).save(f, format="PNG")
            output["outputs"][key] = {"uri": p.resolve().as_uri(), "sha256": sha(p)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            created.append(args.output)
            f.write(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    except Exception:
        for p in created:
            p.unlink()  # only exclusively-created files of this invocation
        raise
    print(json.dumps({"status": "completed", "task": args.task, "output": str(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2) from None
