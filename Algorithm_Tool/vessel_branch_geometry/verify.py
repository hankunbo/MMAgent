"""Deterministic CPU checks; synthetic by default, existing case only when explicit."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import line
from skimage.morphology import binary_dilation, disk

from Algorithm_Tool.vessel_branch_geometry import runner as g


def y_mask(angle, length=130):
    a = np.zeros((384, 384), bool)
    pairs = [((192, 318), (192, 192))]
    for sign in (-1, 1):
        q = (
            round(192 + sign * length * math.sin(math.radians(angle / 2))),
            round(192 - length * math.cos(math.radians(angle / 2))),
        )
        pairs.append(((192, 192), q))
    for p, q in pairs:
        yy, xx = line(p[1], p[0], q[1], q[0])
        a[yy, xx] = True
    return binary_dilation(a, disk(2)), [192.0, 342.0], 24.0


def rotation(m, c, k):
    h, w = m.shape
    x, y = c
    for _ in range(k):
        x, y, h, w = y, w - 1 - x, w, h
    return np.rot90(m, k), [x, y]


def exercise():
    checks, results = {}, {}
    for degree in [60, 90, 120]:
        m, c, r = y_mask(degree)
        result = g.measure(m, c, r)
        results[f"y{degree}"] = result
        checks[f"known_y_{degree}"] = (
            result["count"] == 1 and abs(result["median_degrees"] - degree) <= 5
        )
        if result["count"]:
            row = next(x for x in result["junctions"] if x["status"] == "measured")
            checks[f"parent_child_{degree}"] = row["parent_samples_xy"][-1][1] > 192 and all(
                points[-1][1] < 192 for points in row["daughter_samples_xy"]
            )
    m, c, r = y_mask(90)
    for k in [1, 2, 3]:
        a, cc = rotation(m, c, k)
        b = g.measure(a, cc, r)
        checks[f"rotation_{k}"] = b["count"] == 1 and abs(b["median_degrees"] - 90) < 5
        if b["count"]:
            row = next(x for x in b["junctions"] if x["status"] == "measured")
            expected = rotation(m, [192, 192], k)[1]
            checks[f"coordinate_restore_{k}"] = (
                np.linalg.norm(np.array(row["vertex_xy"]) - expected) < 3
            )
    enlarged = np.repeat(np.repeat(m, 2, 0), 2, 1)
    scaled = g.measure(enlarged, (np.array(c) * 2 + 0.5).tolist(), r * 2)
    checks["scale_2"] = (
        scaled["count"] == 1
        and abs(scaled["median_degrees"] - results["y90"]["median_degrees"]) < 1e-10
    )
    short = g.measure(*y_mask(90, 20))
    checks["short_y_scale_aware"] = short["count"] == 1 and abs(short["median_degrees"] - 90) < 5
    x = np.zeros_like(m)
    for p, q in [((70, 70), (314, 314)), ((70, 314), (314, 70))]:
        yy, xx = line(p[1], p[0], q[1], q[0])
        x[yy, xx] = True
    x = binary_dilation(x, disk(2))
    crossing = g.measure(x, [70, 342], r)
    results["crossing"] = crossing
    checks["crossing_excluded"] = crossing["count"] == 0 and any(
        j["reason"] == "crossing_candidate" for j in crossing["junctions"]
    )
    unknown = g.measure(m, [30, 30], r)
    checks["unrooted_rejected"] = unknown["count"] == 0
    loop = m.copy()
    loop[96:103, 100:285] = True
    loop_result = g.measure(loop, c, r)
    results["loop"] = loop_result
    checks["loop_rejected"] = loop_result["count"] == 0
    empty = g.measure(np.zeros((128, 128), bool), [64, 64], 10.0)
    checks["empty_null_not_zero"] = (
        empty["count"] == 0 and empty["median_degrees"] is None and empty["mean_degrees"] is None
    )
    checks["deterministic_repeat"] = g.measure(m, c, r) == g.measure(m, c, r)
    # Unit fault injection for scale matching, separately from the analytic Y
    # measurements above. No fixture is presented as a real vessel observation.
    from Algorithm_Tool.vessel_branch_geometry.stability import checked_measure

    for failure in ["missing", "ambiguous", "angle_drift"]:
        calls = []
        raw = copy.deepcopy(results["y90"])
        original = next(j for j in raw["junctions"] if j["status"] == "measured")

        def sample(mm, cc, rr, *, calls=calls, raw=raw, original=original, failure=failure):
            calls.append(None)
            if len(calls) == 1:
                return raw
            scale = g.POLICY["scale_check_factors"][len(calls) - 2]
            point = (np.array(original["vertex_xy"]) + 0.5) * scale - 0.5
            item = {
                "status": "measured",
                "vertex_xy": point.tolist(),
                "angle_degrees": original["angle_degrees"] + (6 if failure == "angle_drift" else 0),
            }
            return {
                "junctions": []
                if failure == "missing"
                else [item] * (2 if failure == "ambiguous" else 1)
            }

        result = checked_measure(
            sample, lambda mm, cc, rr: (mm, cc, None, mm.shape, 0), m, c, r, g.POLICY
        )
        checks["scale_qc_" + failure] = result["count"] == 0 and any(
            j["reason"] == "scale_topology_or_angle_unstable"
            and j["candidate_angle_degrees"] is not None
            for j in result["junctions"]
        )
    for name, a, cc, rr in [
        ("float_mask", m.astype(float), c, r),
        ("nonbinary", m.astype(int) * 2, c, r),
        ("rgb", np.zeros((10, 10, 3), dtype="uint8"), [5, 5], 4),
        ("nan_center", m, [float("nan"), 20], r),
        ("infinite_radius", m, c, float("inf")),
        ("outside_center", m, [-1, 0], r),
        ("tiny_radius", m, c, 1),
        ("large_mask", np.zeros((2049, 2049), bool), [50, 50], r),
    ]:
        try:
            g.measure(a, cc, rr)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    return checks, results


def file_inputs(root):
    checks = {}
    for name, mode, size, value in [
        ("png_rgb", "RGB", (4, 4), (0, 0, 0)),
        ("png_nonbinary", "L", (4, 4), 2),
        ("png_large", "L", (2049, 2049), 0),
    ]:
        path = root / (name + ".png")
        Image.new(mode, size, value).save(path)
        try:
            g.read_mask(path)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    path = root / "valid-binary.png"
    Image.new("L", (16, 16), 255).save(path)
    mask, binding = g.read_mask(path)
    checks["png_binding"] = (
        mask.all()
        and binding["sha256"] == g.digest(path.read_bytes())
        and binding["dimensions_hw"] == [16, 16]
    )
    return checks


def cases(output):
    baseline = Path("/opt/data/test/PVBM-reproduction/results-v1")
    results = {}
    for path in sorted(baseline.glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            results[path.stem] = g.measure(
                source["mask"], source["center"].tolist(), float(source["radius"])
            )
    original = results["case_outside"]
    tests = {}
    for name in ["case_repeat", "case_rot90", "case_rot180", "case_rot270", "case_scale2"]:
        r = results[name]
        expected = sorted(
            j["angle_degrees"] for j in original["junctions"] if j["status"] == "measured"
        )
        actual = sorted(j["angle_degrees"] for j in r["junctions"] if j["status"] == "measured")
        tests[name] = r["count"] == original["count"] and np.allclose(
            actual, expected, rtol=0, atol=1e-10
        )
    disputed = [
        j
        for j in original["junctions"]
        if np.linalg.norm(np.array(j["vertex_xy"]) - [282, 515]) < 5
    ]
    tests["old_disputed_point_excluded"] = bool(disputed) and all(
        j["status"] == "excluded" for j in disputed
    )
    candidates = [j for j in original["junctions"] if j.get("scale_checks")]
    tests["case_scale_decisions_match_summary"] = bool(candidates) and all(
        (j["status"] == "measured") == all(s["passed"] for s in j["scale_checks"])
        for j in candidates
    )
    # Also test the public runner's full mask input, not only PVBM's old cropped
    # ROI fixtures. The two inputs are separately hashed and never substituted.
    vein_path = Path("/opt/data/test/MMAgent/.local/glaucoma-case-evidence-v1/avod/vein_mask.png")
    disc_path = Path("/opt/data/test/segman-slo-deployment/results/case-single-003/disc.png")
    vein, vein_binding = g.read_mask(vein_path)
    disc, disc_binding = g.read_mask(disc_path)
    dy, dx = np.nonzero(disc)
    center, radius = [float(dx.mean()), float(dy.mean())], math.sqrt(disc.sum() / math.pi)
    full = results["full_mask"] = g.measure(vein, center, radius)
    expected = sorted(j["angle_degrees"] for j in full["junctions"] if j["status"] == "measured")
    baseline_angles = sorted(
        j["angle_degrees"] for j in original["junctions"] if j["status"] == "measured"
    )
    tests["full_mask_and_existing_roi_agree"] = full["count"] == original["count"] and np.allclose(
        expected, baseline_angles, rtol=0, atol=1e-10
    )
    for k in [1, 2, 3]:
        rotated, cc = rotation(vein, center, k)
        measured = results[f"full_mask_rot{k}"] = g.measure(rotated, cc, radius)
        actual = sorted(
            j["angle_degrees"] for j in measured["junctions"] if j["status"] == "measured"
        )
        tests[f"full_mask_rotation_{k}"] = measured["count"] == full["count"] and np.allclose(
            actual, expected, rtol=0, atol=1e-10
        )
    enlarged = np.repeat(np.repeat(vein, 2, 0), 2, 1)
    measured = results["full_mask_scale2"] = g.measure(
        enlarged, (np.array(center) * 2 + 0.5).tolist(), radius * 2
    )
    actual = sorted(j["angle_degrees"] for j in measured["junctions"] if j["status"] == "measured")
    tests["full_mask_scale2"] = measured["count"] == full["count"] and np.allclose(
        actual, expected, rtol=0, atol=1e-10
    )
    (output / "full-mask-bindings.json").write_text(
        json.dumps({"vein_mask": vein_binding, "disc_mask": disc_binding}, indent=2)
    )
    image = np.asarray(Image.open(baseline / "input.png"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image, cmap="gray")
    for j in original["junctions"]:
        x, y = j["vertex_xy"]
        ok = j["status"] == "measured"
        ax.plot(x, y, "o", color="lime" if ok else "orange", markersize=4)
        ax.text(x + 3, y, j["junction_id"], color="lime" if ok else "orange", fontsize=7)
        if ok:
            for group, color in [("parent_samples_xy", "cyan"), ("daughter_samples_xy", "yellow")]:
                lines = [j[group]] if group.startswith("parent") else j[group]
                for points in lines:
                    a, b = np.array(points)
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2)
            ax.text(x + 5, y + 14, f"{j['angle_degrees']:.2f} deg", color="yellow", fontsize=9)
    ax.set_title("Green: measured; orange: excluded; cyan: parent; yellow: daughters")
    fig.tight_layout()
    fig.savefig(output / "case-overlay.png", dpi=130)
    plt.close(fig)
    (output / "baseline-input-sha256.json").write_text(
        json.dumps(
            {str(p): g.digest(p.read_bytes()) for p in sorted(baseline.glob("*.npz"))}, indent=2
        )
    )
    return tests, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-case", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, results = exercise()
    checks.update(file_inputs(args.output))
    if args.include_case:
        case_checks, case_results = cases(args.output)
        checks.update(case_checks)
        results.update(case_results)
    checks = {k: bool(v) for k, v in checks.items()}
    report = {
        "checks": checks,
        "results": results,
        "policy": g.POLICY,
        "status": "passed" if all(checks.values()) else "failed",
        "real_angle_ground_truth_evaluated": False,
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
