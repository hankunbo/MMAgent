"""Versioned soft mask QC. Geometry defects are weights, not case-stop signals.

The factor is an engineering heuristic, NOT calibrated segmentation accuracy.
Malformed bytes, shape/scope/checksum conflicts remain technical errors.
"""

from packages.grounding.models import digest
from packages.tool_execution.qc import grayscale_png

from .case_qc import components

POLICY = {
    "version": "mask-quality-weight/1.1.0",
    "formula": "disc_largest_fraction * cup_largest_fraction * cup_containment * span_stability",
    "zero": "missing_measurement_not_negative_disease_evidence",
    "masks": "original_unmodified_all_foreground_pixels",
    "calibration": "engineering_heuristic_not_accuracy",
}


def spans(mask, shape):
    """Inspect components for sensitivity only; never edit the measured mask."""
    h, w = shape
    remaining = {i for i, v in enumerate(mask) if v}
    all_rows = [i // w for i in remaining]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            y, x = divmod(i, w)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    j = yy * w + xx
                    if 0 <= yy < h and 0 <= xx < w and j in remaining:
                        remaining.remove(j)
                        stack.append(j)
        groups.append(group)
    if not groups:
        return dict(all=0, main=0)
    main = max(groups, key=len)
    rows = [i // w for i in main]
    return dict(all=max(all_rows) - min(all_rows) + 1, main=max(rows) - min(rows) + 1)


def mask_quality(disc_png, cup_png):
    shape, disc = grayscale_png(disc_png, 1048576)
    cup_shape, cup = grayscale_png(cup_png, 1048576)
    if shape != cup_shape:
        raise ValueError("quality_mask_shape_mismatch")
    counts = dict(disc=components(disc, shape), cup=components(cup, shape))
    fractions = {k: (v[0] / sum(v) if v else 0.0) for k, v in counts.items()}
    cup_area = sum(bool(v) for v in cup)
    containment = (
        sum(bool(c and d) for c, d in zip(cup, disc, strict=True)) / cup_area if cup_area else 0.0
    )
    ds, cs = spans(disc, shape), spans(cup, shape)
    whole = cs["all"] / ds["all"] if ds["all"] else None
    main = cs["main"] / ds["main"] if ds["main"] else None
    stability = min(whole, main) / max(whole, main) if whole and main else 0.0
    q = fractions["disc"] * fractions["cup"] * containment * stability
    reasons = [
        ("empty_" if not values else "disconnected_") + key
        for key, values in counts.items()
        if len(values) != 1
    ]
    if cup_area and containment < 1:
        reasons.append("cup_outside_disc")
    if stability < 1 and cup_area:
        reasons.append("bbox_sensitive_to_secondary_components")
    computable = bool(counts["disc"] and counts["cup"])
    return dict(
        schema_version="measurement-quality/1.0.0",
        policy=POLICY,
        policy_checksum=digest(POLICY),
        connected_component_pixel_counts=counts,
        components=dict(
            disc_concentration=fractions["disc"],
            cup_concentration=fractions["cup"],
            cup_containment=containment,
            span_stability=stability,
        ),
        sensitivity=dict(
            all_foreground_vcdr=whole if cup_area else None,
            largest_components_vcdr=main if cup_area else None,
            role="quality_diagnostic_not_replacement_measurement",
        ),
        quality_factor=q,
        measurement_available=computable,
        availability_score=float(computable),
        unresolved_fraction=1.0 - q,
        reasons=reasons,
        masks_modified=False,
        stop_case=False,
        next_action="measure_with_quality"
        if computable
        else "bounded_alternative_or_visual_observation",
        missing_disease_direction=None,
        mask_checksums=dict(disc=digest(list(disc)), cup=digest(list(cup))),
    )
