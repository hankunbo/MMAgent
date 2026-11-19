"""Portable output contract; no scientific/model imports at public import time."""

import hashlib
import json
import math

VERSION = "av-vascular-metrics/1.0.0"
POLICY = {
    "version": VERSION,
    "population": "segmented_visible_vessels_not_capillaries",
    "overlap": "exclude_A_and_V_overlap_from_both_classes",
    "skeleton": "skimage_Zhang_native_pixel_grid",
    "density": "skeleton_pixel_count / explicit_ROI_pixel_count; unit pixel^-1",
    "area_density": "vessel_pixel_count / explicit_ROI_pixel_count",
    "width": "vessel_pixel_count / skeleton_pixel_count; pixel proxy, not CRAE/CRVE",
    "normalized_width": "width_proxy / sqrt(4*disc_area/pi)",
    "annulus": "disc centroid radius [1.0D,1.5D), D=equivalent disc diameter",
    "tortuosity": "8-neighbour graph without diagonal shortcut; Euclidean arc/chord",
    "minimum_segment": "max(10 native pixels, 0.10D); endpoints/junctions delimit segments",
    "fractal": "occupied boxes on skeleton; sizes 4,8,...<=min(H,W)/4; fixed top-left grid",
    "fractal_fit": "OLS log(N) vs log(1/size); report R2 and every count; minimum 3 scales",
    "physical_calibration": "absent; no micron conversion or population cutoff",
}


def checksum(value):
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate(value):
    required = {"schema_version", "policy", "policy_checksum", "input_bindings", "disc",
                "overlap_pixels", "classes", "research_only", "checksum"}
    if set(value) != required or value["schema_version"] != VERSION:
        raise ValueError("av_output_schema")
    if value["policy"] != POLICY or value["policy_checksum"] != checksum(POLICY):
        raise ValueError("av_policy_drift")
    if value["research_only"] is not True or value["checksum"] != checksum(
        {k: v for k, v in value.items() if k != "checksum"}
    ):
        raise ValueError("av_output_checksum")
    bindings = value["input_bindings"]
    if set(bindings) != {"artery_mask", "vein_mask", "disc_mask"}:
        raise ValueError("av_input_ports")
    shapes = []
    for item in bindings.values():
        if set(item) != {"sha256", "bytes", "dimensions_hw"}:
            raise ValueError("av_input_binding")
        sha = item["sha256"]
        if not isinstance(sha, str) or len(sha) != 71 or not sha.startswith("sha256:"):
            raise ValueError("av_input_checksum")
        if any(c not in "0123456789abcdef" for c in sha[7:]):
            raise ValueError("av_input_checksum")
        shape = item["dimensions_hw"]
        if len(shape) != 2 or any(type(n) is not int or not 16 <= n <= 4096 for n in shape):
            raise ValueError("av_input_dimensions")
        if math.prod(shape) > 4_000_000 or type(item["bytes"]) is not int or not 0 < item["bytes"] <= 32_000_000:
            raise ValueError("av_input_bound")
        shapes.append(shape)
    if shapes[1:] != shapes[:1] * 2:
        raise ValueError("av_input_alignment")
    h, w = shapes[0]
    disc = value["disc"]
    if set(disc) != {"area_pixels", "diameter_pixels", "center_xy", "annulus_pixels", "annulus_fraction_in_image"}:
        raise ValueError("av_disc_schema")
    if not 16 <= disc["area_pixels"] < h*w/2 or not math.isclose(
        disc["diameter_pixels"], math.sqrt(4*disc["area_pixels"]/math.pi), rel_tol=1e-12
    ) or not 0 <= disc["center_xy"][0] < w or not 0 <= disc["center_xy"][1] < h:
        raise ValueError("av_disc_geometry")
    if not 0 <= value["overlap_pixels"] <= h*w or set(value["classes"]) != {"artery", "vein"}:
        raise ValueError("av_classes")
    for metrics in value["classes"].values():
        if set(metrics) != {"whole_image", "annulus", "tortuosity", "skeleton_fractal"}:
            raise ValueError("av_metric_schema")
        for region, denominator in [("whole_image", h*w), ("annulus", disc["annulus_pixels"])]:
            m = metrics[region]
            if set(m) != {"roi_pixels", "vessel_pixels", "skeleton_pixels", "area_density", "skeleton_density", "width_proxy_pixels", "width_proxy_disc_ratio"}:
                raise ValueError("av_density_schema")
            if m["roi_pixels"] != denominator or not 0 <= m["skeleton_pixels"] <= m["vessel_pixels"] <= denominator:
                raise ValueError("av_density_counts")
            for key, numerator in [("area_density",m["vessel_pixels"]), ("skeleton_density",m["skeleton_pixels"])]:
                if m[key] != (numerator/denominator if denominator else None):
                    raise ValueError("av_density_ratio")
            width = m["vessel_pixels"]/m["skeleton_pixels"] if m["skeleton_pixels"] else None
            if m["width_proxy_pixels"] != width or m["width_proxy_disc_ratio"] != (
                width/disc["diameter_pixels"] if width is not None else None
            ):
                raise ValueError("av_width_ratio")
        t = metrics["tortuosity"]
        if set(t) != {"count", "excluded_short", "excluded_closed", "minimum_length_pixels", "mean", "median", "min", "max", "segments"} or t["count"] != len(t["segments"]):
            raise ValueError("av_tortuosity_schema")
        if t["count"] > 10000 or t["minimum_length_pixels"] != max(10.0, .1*disc["diameter_pixels"]):
            raise ValueError("av_tortuosity_bound")
        vals = []
        for s in t["segments"]:
            if set(s) != {"arc_pixels", "chord_pixels", "ratio"} or not 0 < s["chord_pixels"] <= s["arc_pixels"] + 1e-9:
                raise ValueError("av_tortuosity_geometry")
            if s["arc_pixels"] < t["minimum_length_pixels"] or not math.isclose(s["ratio"], s["arc_pixels"]/s["chord_pixels"], rel_tol=1e-12):
                raise ValueError("av_tortuosity_ratio")
            vals.append(s["ratio"])
        import statistics

        for field, expected in {
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
        }.items():
            if t[field] != expected:
                raise ValueError("av_tortuosity_summary")
        f = metrics["skeleton_fractal"]
        if set(f) != {"value", "r_squared", "boxes", "reason"} or len(f["boxes"]) > 12:
            raise ValueError("av_fractal_schema")
        if f["value"] is None:
            if f["r_squared"] is not None or f["reason"] != "fewer_than_three_occupied_scales":
                raise ValueError("av_fractal_missing")
        elif not 0 <= f["value"] <= 2.1 or not 0 <= f["r_squared"] <= 1.0000001 or f["reason"] is not None:
            raise ValueError("av_fractal_range")
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("av_nonfinite")
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        elif isinstance(x, list):
            for v in x:
                finite(v)
    finite(value)
    return value
