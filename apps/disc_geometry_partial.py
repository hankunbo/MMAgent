"""Partial disc-centred ray geometry over unchanged masks; no medical judgment.

This is a separate measurement definition from the strict full-mask v1 tool.
Missing cup intersections or disconnected ray intervals never become zero width.
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

from apps import disc_geometry as original

VERSION = "partial-disc-rim-geometry/1.0.0"
METHOD = "disc_centroid_single_interval_partial_rays/1.0.0"


def _mask(shape, pixels):
    try:
        center, positions = original.validate_mask(shape, pixels)
        return center, positions, None
    except ValueError as error:
        if str(error) != "mask_has_holes":
            raise
        # Only this particular topology condition becomes a per-ray limitation.
        # Shape, binary values, clipping, size and disconnected masks still fail.
        positions = [i for i, value in enumerate(pixels) if value]
        w = shape[1]
        center = fmean(i % w for i in positions), fmean(i // w for i in positions)
        return center, positions, "mask_has_holes"


def rim(shape, disc, cup):
    (cx, cy), positions, disc_condition = _mask(shape, disc)
    _, _, cup_condition = _mask(shape, cup)
    if any(c and not d for c, d in zip(cup, disc, strict=True)):
        raise ValueError("cup_outside_disc")
    h, w = shape
    extent = max(math.hypot(i % w - cx, i // w - cy) for i in positions) + 2
    steps = math.ceil(extent / 0.25)

    def spans(mask, angle):
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        intervals, start = [], None
        for j in range(steps + 1):
            r = j * 0.25
            x, y = round(cx + r * dx), round(cy + r * dy)
            inside = 0 <= x < w and 0 <= y < h and bool(mask[y * w + x])
            if inside and start is None:
                start = r
            elif not inside and start is not None:
                intervals.append((start, r))
                start = None
        if start is not None:
            raise ValueError("partial_rim_sampling_extent_incomplete")
        return intervals

    outer = [spans(disc, a) for a in range(360)]
    rows = []
    for angle in range(360):
        d, opposite, c = outer[angle], outer[(angle + 180) % 360], spans(cup, angle)
        reason = (
            "disc_axis_not_single_contiguous_interval"
            if len(d) != 1 or len(opposite) != 1 or d[0][0] != 0 or opposite[0][0] != 0
            else "cup_not_intersected"
            if not c
            else "cup_has_multiple_intervals"
            if len(c) != 1
            else "cup_boundary_outside_disc"
            if c[0][1] > d[0][1]
            else None
        )
        width = d[0][1] - c[0][1] if reason is None else None
        diameter = d[0][1] + opposite[0][1] if reason is None else None
        rows.append(
            dict(
                angle_degrees=angle,
                rim_width_px=width,
                disc_diameter_px=diameter,
                width_over_same_axis_disc_diameter=width / diameter if reason is None else None,
                cup_interval_px=list(c[0]) if len(c) == 1 else None,
                missing_reason=reason,
            )
        )
    valid = [r for r in rows if r["missing_reason"] is None]
    selected = min(
        valid,
        key=lambda r: (r["width_over_same_axis_disc_diameter"], r["angle_degrees"]),
        default=None,
    )
    full = len(valid) == 360
    return dict(
        method=METHOD,
        status="full_geometry" if full else "partial_geometry" if valid else "unavailable_geometry",
        center_xy=[cx, cy],
        radial_step_px=0.25,
        angle_step_degrees=1,
        precision="nearest_pixel_raster_estimate_not_subpixel_accuracy",
        orientation="image_only",
        anatomical_comparison=None,
        mask_conditions=dict(disc=disc_condition, cup=cup_condition),
        disc_centroid_inside_cup=bool(cup[round(cy) * w + round(cx)]),
        valid_rays=len(valid),
        invalid_rays=360 - len(valid),
        geometric_direction_coverage=len(valid) / 360,
        coverage_semantics="fraction_of_defined_image_rays_not_probability_or_clinical_quality",
        minimum_available_normalized_width=selected,
        full_circumference_minimum=selected if full else None,
        full_circumference_minimum_unavailable_reason=None
        if full
        else "incomplete_geometry_do_not_extrapolate_unmeasured_directions",
        missing_direction_reasons=dict(
            Counter(r["missing_reason"] for r in rows if r["missing_reason"])
        ),
        masks_modified=False,
        clinical_threshold=None,
        clinical_effect=None,
        rays=rows,
    )


def measure(disc_path, cup_path):
    shape, disc, disc_ref = original.read_mask(disc_path)
    cup_shape, cup, cup_ref = original.read_mask(cup_path)
    if shape != cup_shape:
        raise ValueError("mask_shape_mismatch")
    return dict(
        schema_version=VERSION,
        task="partial_rim",
        shape_hw=list(shape),
        inputs=dict(disc_mask=disc_ref, cup_mask=cup_ref),
        correlation_group="disc_mask:sha256:" + disc_ref["sha256"],
        coordinate_assumption="same_original_image_grid; square_pixels; no_resize_or_warp",
        binding_scope=(
            "content_geometry_only; original_Runtime_must_verify_case_and_current_qualification"
        ),
        use="research_measurement",
        disease_probability=None,
        measurement=rim(shape, disc, cup),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disc-mask", type=Path, required=True)
    parser.add_argument("--cup-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original.write_result(args.output, measure(args.disc_mask, args.cup_mask))


if __name__ == "__main__":
    main()
