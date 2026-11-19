"""Per-junction resampling QC; excluded candidates remain in the result."""

import numpy as np


def checked_measure(measure_once, normalize, mask, center, radius, policy):
    result = measure_once(mask, center, radius)
    candidates = [j for j in result["junctions"] if j["status"] == "measured"]
    if not candidates:
        return result
    # Perturb a fixed disc-normalized grid, not the caller's pixel resolution.
    # Upsampling a mask must not change which candidates pass this quality check.
    # This tests raster stability; it does not prove segmentation accuracy.
    original_h, original_w = mask.shape
    mask, center, _, _, turns = normalize(mask, center, radius)
    radius = policy["reference_disc_diameter"] / 2
    h, w = mask.shape
    positions = {}
    for row in candidates:
        x, y = row["vertex_xy"]
        hh, ww = original_h, original_w
        for _ in range(turns):
            x, y, hh, ww = y, ww - 1 - x, ww, hh
        positions[row["junction_id"]] = (np.array([x, y]) + 0.5) * [w / ww, h / hh] - 0.5
    for row in candidates:
        row["scale_checks"] = []
    for scale in policy["scale_check_factors"]:
        nh, nw = round(h * scale), round(w * scale)
        others = []
        if nh * nw <= policy["maximum_pixels"]:
            yy = np.minimum(((np.arange(nh) + 0.5) * h / nh).astype(int), h - 1)
            xx = np.minimum(((np.arange(nw) + 0.5) * w / nw).astype(int), w - 1)
            cc = (np.array(center) + 0.5) * [nw / w, nh / h] - 0.5
            checked = measure_once(mask[np.ix_(yy, xx)], cc.tolist(), radius * scale)
            others = [j for j in checked["junctions"] if j["status"] == "measured"]
        for row in candidates:
            matches = [
                j
                for j in others
                if np.linalg.norm(
                    (np.array(j["vertex_xy"]) + 0.5) * [w / nw, h / nh]
                    - 0.5
                    - positions[row["junction_id"]]
                )
                <= 2 * radius * policy["scale_match_disc_diameters"]
            ]
            if len(matches) == 1:
                point = (np.array(matches[0]["vertex_xy"]) + 0.5) * [w / nw, h / nh] - 0.5
                if (
                    sum(
                        np.linalg.norm(point - p)
                        <= 2 * radius * policy["scale_match_disc_diameters"]
                        for p in positions.values()
                    )
                    != 1
                ):
                    matches = []
            degree = matches[0]["angle_degrees"] if len(matches) == 1 else None
            passed = (
                degree is not None
                and abs(degree - row["angle_degrees"]) <= policy["scale_angle_tolerance_degrees"]
            )
            row["scale_checks"].append(
                {"scale": scale, "angle_degrees": degree, "passed": bool(passed)}
            )
    for row in candidates:
        if not all(s["passed"] for s in row["scale_checks"]):
            row.update(
                status="excluded",
                reason="scale_topology_or_angle_unstable",
                candidate_angle_degrees=row["angle_degrees"],
                angle_degrees=None,
            )
    values = [j["angle_degrees"] for j in result["junctions"] if j["status"] == "measured"]
    result.update(
        count=len(values),
        median_degrees=float(np.median(values)) if values else None,
        mean_degrees=float(np.mean(values)) if values else None,
        status="measured" if values else "no_valid_bifurcation",
    )
    return result
