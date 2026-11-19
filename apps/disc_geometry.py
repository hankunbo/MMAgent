"""CPU geometry on existing, aligned SLO masks; no model or diagnostic scoring."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path
from statistics import fmean

from packages.tool_execution.qc import grayscale_png

VERSION = "disc-geometry/1.0.0"
MAX_PIXELS = 4_194_304
MAX_BYTES = 8_388_608
ORIENTATIONS = ("image_only", "superior_up_nasal_left", "superior_up_nasal_right")


def _connected(pixels, h, w, value, start):
    seen = bytearray(h * w)
    seen[start] = 1
    queue = deque([start])
    count = 0
    while queue:
        i = queue.popleft()
        count += 1
        y, x = divmod(i, w)
        for j in (
            i - w if y else -1,
            i + w if y + 1 < h else -1,
            i - 1 if x else -1,
            i + 1 if x + 1 < w else -1,
        ):
            if j >= 0 and not seen[j] and pixels[j] == value:
                seen[j] = 1
                queue.append(j)
    return count


def validate_mask(shape, pixels):
    if (
        len(shape) != 2
        or any(type(v) is not int or v < 3 for v in shape)
        or math.prod(shape) > MAX_PIXELS
        or len(pixels) != math.prod(shape)
        or not set(pixels) <= {0, 255}
    ):
        raise ValueError("mask_binary_shape_invalid")
    h, w = shape
    positions = [i for i, value in enumerate(pixels) if value]
    if len(positions) < 5:
        raise ValueError("mask_empty_or_too_small")
    xs, ys = [i % w for i in positions], [i // w for i in positions]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1:
        raise ValueError("mask_clipped_at_image_boundary")
    if _connected(pixels, h, w, 255, positions[0]) != len(positions):
        raise ValueError("mask_multiple_components")
    # Check holes in a small padded bounding box, never fill/repair a prediction.
    bw, bh = x1 - x0 + 3, y1 - y0 + 3
    box = b"".join(bytes(pixels[y * w + x0 - 1 : y * w + x1 + 2]) for y in range(y0 - 1, y1 + 2))
    if _connected(box, bh, bw, 0, 0) != len(box) - len(positions):
        raise ValueError("mask_has_holes")
    return (fmean(xs), fmean(ys)), positions


def ellipse(shape, pixels):
    (cx, cy), positions = validate_mask(shape, pixels)
    w = shape[1]
    xx = fmean((i % w - cx) ** 2 for i in positions)
    yy = fmean((i // w - cy) ** 2 for i in positions)
    xy = fmean((i % w - cx) * (i // w - cy) for i in positions)
    gap = math.hypot(xx - yy, 2 * xy)
    major_eigen, minor_eigen = (xx + yy + gap) / 2, (xx + yy - gap) / 2
    if minor_eigen <= 0:
        raise ValueError("mask_degenerate_ellipse")
    major, minor = 4 * math.sqrt(major_eigen), 4 * math.sqrt(minor_eigen)
    isotropic = gap / major_eigen <= 1e-6
    angle = None if isotropic else math.degrees(math.atan2(2 * xy, xx - yy)) / 2 % 180
    return {
        "method": "foreground_pixel_centres_second_moment_equivalent_ellipse",
        "center_xy": [cx, cy],
        "area_px": len(positions),
        "covariance_xy_px2": [[xx, xy], [xy, yy]],
        "major_axis_px": major,
        "minor_axis_px": minor,
        "minor_over_major": minor / major,
        "major_over_minor": major / minor,
        "eccentricity": math.sqrt(max(0, 1 - minor_eigen / major_eigen)),
        "major_axis_angle_degrees": angle,
        "angle_missing_reason": "near_circular_axis_not_identifiable" if isotropic else None,
        "angle_convention": "image x right, y down; clockwise from x; modulo 180",
        "isotropy_numerical_tolerance": 1e-6,
        "clinical_tilt_angle": None,
        "clinical_tilt_angle_missing_reason": "2D_mask_does_not_measure_3D_tilt",
    }


def _percentile(values, p):
    ordered = sorted(values)
    at = (len(ordered) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def rim(shape, disc, cup, orientation="image_only", orientation_source=None):
    if (
        orientation not in ORIENTATIONS
        or ((orientation == "image_only") != (orientation_source is None))
        or (
            orientation_source is not None
            and (
                not isinstance(orientation_source, str)
                or not orientation_source.strip()
                or len(orientation_source) > 256
            )
        )
    ):
        raise ValueError("orientation_definition_invalid")
    (cx, cy), positions = validate_mask(shape, disc)
    validate_mask(shape, cup)
    h, w = shape
    if any(c and not d for c, d in zip(cup, disc, strict=True)):
        raise ValueError("cup_outside_disc")
    if not cup[round(cy) * w + round(cx)]:
        raise ValueError("disc_centroid_outside_cup")
    extent = max(math.hypot(i % w - cx, i // w - cy) for i in positions) + 2
    steps = math.ceil(extent / 0.25)

    def radius(mask, angle):
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        first = None
        for j in range(1, steps + 1):
            r = j * 0.25
            x, y = round(cx + r * dx), round(cy + r * dy)
            inside = 0 <= x < w and 0 <= y < h and mask[y * w + x] != 0
            if not inside and first is None:
                first = r
            elif inside and first is not None:
                return None  # Multiple boundary crossings: no invented width.
        return first

    outer = [radius(disc, a) for a in range(360)]
    inner = [radius(cup, a) for a in range(360)]
    rows = []
    for a in range(360):
        r, c, opposite = outer[a], inner[a], outer[(a + 180) % 360]
        valid = all(v is not None for v in (r, c, opposite))
        rows.append(
            {
                "angle_degrees": a,
                "rim_width_px": r - c if valid else None,
                "disc_diameter_px": r + opposite if valid else None,
                "width_over_same_axis_disc_diameter": (r - c) / (r + opposite) if valid else None,
                "missing_reason": None if valid else "multiple_boundary_crossings",
            }
        )
    valid = [row for row in rows if row["missing_reason"] is None]
    if not valid:
        raise ValueError("no_radially_simple_measurements")
    sectors = {}
    for label, center in (("right", 0), ("down", 90), ("left", 180), ("up", 270)):
        selected = [r for r in valid if -45 <= (r["angle_degrees"] - center + 180) % 360 - 180 < 45]
        widths = [r["rim_width_px"] for r in selected]
        sectors[label] = {
            "center_angle_degrees": center,
            "sector_width_degrees": 90,
            "valid_rays": len(widths),
            "expected_rays": 90,
            "mean_width_px": fmean(widths) if widths else None,
            "p05_width_px": _percentile(widths, 0.05) if widths else None,
            "median_width_px": _percentile(widths, 0.5) if widths else None,
            "p95_width_px": _percentile(widths, 0.95) if widths else None,
        }
    anatomical = None
    anatomical_reason = "anatomical_orientation_not_provided"
    if orientation != "image_only":
        nasal, temporal = (180, 0) if orientation.endswith("left") else (0, 180)
        widths = {
            k: rows[a]["rim_width_px"]
            for k, a in (("I", 90), ("S", 270), ("N", nasal), ("T", temporal))
        }
        if all(v is not None for v in widths.values()):
            anatomical = {
                "cardinal_widths_px": widths,
                "ISNT_nonincreasing": widths["I"] >= widths["S"] >= widths["N"] >= widths["T"],
                "comparison": "I>=S>=N>=T; raw widths, not sector means or normalized ratios",
                "method": "mask_cardinal_ray_estimate_not_stereoscopic_expert_grading",
            }
            anatomical_reason = None
        else:
            anatomical_reason = "cardinal_ray_has_multiple_boundary_crossings"
    return {
        "method": "disc_centroid_nearest_pixel_ray_sampling",
        "center_xy": [cx, cy],
        "radial_step_px": 0.25,
        "angle_step_degrees": 1,
        "direction": "0=image_right;90=image_down;180=image_left;270=image_up",
        "precision": "raster_estimate_not_subpixel_accuracy",
        "rays": rows,
        "valid_rays": len(valid),
        "invalid_rays": 360 - len(valid),
        "minimum_width": min(valid, key=lambda r: (r["rim_width_px"], r["angle_degrees"])),
        "minimum_normalized_width": min(
            valid, key=lambda r: (r["width_over_same_axis_disc_diameter"], r["angle_degrees"])
        ),
        "image_sectors": sectors,
        "orientation": orientation,
        "orientation_source": orientation_source,
        "anatomical_comparison": anatomical,
        "anatomical_missing_reason": anatomical_reason,
    }


def read_mask(path):
    with Path(path).open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("mask_file_too_large")
    shape, pixels = grayscale_png(data, MAX_PIXELS)
    return shape, pixels, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def measure(task, disc_path, cup_path=None, orientation="image_only", orientation_source=None):
    shape, disc, disc_ref = read_mask(disc_path)
    inputs = {"disc_mask": disc_ref}
    if task == "rim":
        if cup_path is None:
            raise ValueError("cup_mask_required")
        cup_shape, cup, inputs["cup_mask"] = read_mask(cup_path)
        if cup_shape != shape:
            raise ValueError("mask_shape_mismatch")
        result = rim(shape, disc, cup, orientation, orientation_source)
    elif task == "ellipse":
        if cup_path is not None or orientation != "image_only" or orientation_source is not None:
            raise ValueError("ellipse_extra_inputs")
        result = ellipse(shape, disc)
    else:
        raise ValueError("unknown_geometry_task")
    return {
        "schema_version": VERSION,
        "task": task,
        "shape_hw": list(shape),
        "inputs": inputs,
        "correlation_group": "disc_mask:sha256:" + disc_ref["sha256"],
        "coordinate_assumption": "same_original_image_grid; square_pixels; no_resize_or_warp",
        "use": "research_measurement",
        "clinical_effect": None,
        "disease_probability": None,
        "measurement": result,
    }


def write_result(path, result):
    """Publish complete JSON exclusively; a failed commit leaves no partial result."""
    path = Path(path)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".disc-geometry-", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(encoded + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)  # Atomic no-replace; no existing output/input is overwritten.
    finally:
        if temporary is not None:
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("rim", "ellipse"))
    parser.add_argument("--disc-mask", type=Path, required=True)
    parser.add_argument("--cup-mask", type=Path)
    parser.add_argument("--orientation", choices=ORIENTATIONS, default="image_only")
    parser.add_argument("--orientation-source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(
        args.task, args.disc_mask, args.cup_mask, args.orientation, args.orientation_source
    )
    write_result(args.output, result)


if __name__ == "__main__":
    main()
