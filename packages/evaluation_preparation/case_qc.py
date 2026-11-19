"""Mask topology quality gate, not a new disease threshold or mask correction."""

from packages.tool_execution.qc import grayscale_png


def components(mask, shape):
    h, w = shape
    remaining = {i for i, value in enumerate(mask) if value}
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        count = 0
        while stack:
            index = stack.pop()
            count += 1
            y, x = divmod(index, w)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    target = yy * w + xx
                    if 0 <= yy < h and 0 <= xx < w and target in remaining:
                        remaining.remove(target)
                        stack.append(target)
        sizes.append(count)
    return sorted(sizes, reverse=True)


def geometry_quality(disc_png, cup_png):
    shape, disc = grayscale_png(disc_png, 1048576)
    cup_shape, cup = grayscale_png(cup_png, 1048576)
    if shape != cup_shape:
        raise ValueError("new_case_qc_mask_shape")
    counts = dict(disc=components(disc, shape), cup=components(cup, shape))
    reasons = []
    for key, values in counts.items():
        if not values:
            reasons.append("empty_" + key)
        elif len(values) != 1:
            reasons.append("disconnected_" + key)
    if any(c and not d for c, d in zip(cup, disc, strict=True)):
        reasons.append("cup_outside_disc")
    return dict(
        schema_version="new-case-geometry-qc/1.0.0",
        connected_component_pixel_counts=counts,
        bbox_cdr_usable=not reasons,
        reasons=reasons,
        masks_modified=False,
        clinical_conclusion=None,
        policy="single_connected_disc_and_cup_required_for_global_bbox_geometry",
    )
