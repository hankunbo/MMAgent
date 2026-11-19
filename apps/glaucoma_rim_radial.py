"""Reproducible pixel estimate on the already saved cup/disc masks.

This is a case-report calculation, not a new registered clinical tool. Directions
are image axes, not nasal/temporal quadrants. No model inference is performed.
"""

import base64
import math

from packages.object_store import checksum
from packages.tool_execution.qc import grayscale_png


def radial_profile(shape, disc, cup):
    h, w = shape
    if (
        len(disc) != h * w
        or len(cup) != h * w
        or not set(disc) <= {0, 255}
        or not set(cup) <= {0, 255}
    ):
        raise ValueError("rim_binary_shape_invalid")
    positions = [i for i, p in enumerate(disc) if p]
    if not positions or not any(cup) or any(c and not d for c, d in zip(cup, disc, strict=True)):
        raise ValueError("rim_masks_not_nested")
    cx = sum(i % w for i in positions) / len(positions)
    cy = sum(i // w for i in positions) / len(positions)
    if not cup[round(cy) * w + round(cx)]:
        raise ValueError("rim_disc_centroid_outside_cup")
    step = 0.25
    max_steps = math.ceil(math.hypot(h, w) / step) + 1

    def exit_radius(mask, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        for j in range(1, max_steps):
            r = j * step
            x, y = round(cx + r * dx), round(cy + r * dy)
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError("rim_mask_touches_image_edge")
            if not mask[y * w + x]:
                return r
        raise ValueError("rim_boundary_not_found")

    radii = [
        (exit_radius(disc, math.radians(a)), exit_radius(cup, math.radians(a))) for a in range(360)
    ]
    rows = []
    for angle, (outer, inner) in enumerate(radii):
        width = outer - inner
        diameter = outer + radii[(angle + 180) % 360][0]
        rows.append(
            {
                "angle_degrees": angle,
                "rim_width_px": width,
                "disc_diameter_px": diameter,
                "ratio": width / diameter,
            }
        )
    minimum = min(rows, key=lambda r: (r["ratio"], r["angle_degrees"]))
    return {
        "schema": "case-rim-radial-estimate/1.0.0",
        "method": "disc_centroid_first_exit_ray_sampling",
        "definition": "(disc_ray_radius-cup_ray_radius)/(disc_ray_radius+opposite_disc_ray_radius)",
        "center_xy": [cx, cy],
        "shape_hw": list(shape),
        "angle_step_degrees": 1,
        "radial_step_px": step,
        "direction": "0=image_right; 90=image_down; no_anatomical_quadrant_assignment",
        "minimum": minimum,
        "sampled_rays": 360,
        "rays_below_0_1": sum(r["ratio"] < 0.1 for r in rows),
        "axis_examples": [rows[a] for a in (0, 90, 180, 270)],
        "precision": "raster_boundary_estimate_not_subpixel_accuracy",
    }


def from_snapshot(value):
    case = value["case_evidence"]
    masks = {}
    refs = []
    shapes = []
    for name in ("disc", "cup"):
        kind = "ARTIFACT:SLO_SEGMAN_" + name.upper()
        assets = [a for a in case["artifacts"] if a["artifact_type_code"] == kind]
        if len(assets) != 1 or assets[0]["scope"]["case_code"] != case["case_code"]:
            raise ValueError("rim_source_ambiguous_or_scope")
        encoded = case["images"].get("segman_" + name, "")
        prefix = "data:image/png;base64,"
        if not encoded.startswith(prefix) or len(encoded) > 2_800_000:
            raise ValueError("rim_embedded_mask_required")
        body = base64.b64decode(encoded[len(prefix) :], validate=True)
        if checksum(body) != assets[0]["checksum"]:
            raise ValueError("rim_embedded_mask_binding")
        shape, masks[name] = grayscale_png(body, 1_048_576)
        shapes.append(shape)
        refs.append({"code": assets[0]["asset_code"], "checksum": assets[0]["checksum"]})
    if shapes[0] != shapes[1] or list(shapes[0]) != case["input"]["dimensions"][:2]:
        raise ValueError("rim_input_shape_mismatch")
    result = radial_profile(shapes[0], masks["disc"], masks["cup"])
    return {
        **result,
        "case_code": case["case_code"],
        "input_checksum": case["input"]["checksum"],
        "evidence_refs": refs,
    }
