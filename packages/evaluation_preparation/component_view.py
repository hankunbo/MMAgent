"""Explicit connected-component measurement view; original masks stay immutable."""

from packages.dataset_adapters.reading import grayscale_image
from packages.object_store import checksum
from packages.tool_execution.qc import grayscale_png


def select_component(raw):
    shape, pixels = grayscale_png(raw, 1048576)
    if not set(pixels) <= {0, 255}:
        raise ValueError("component_binary_required")
    h, w = shape
    remaining = {i for i, v in enumerate(pixels) if v}
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending, found = [start], [start]
        while pending:
            i = pending.pop()
            y, x = divmod(i, w)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                j = yy * w + xx
                if 0 <= yy < h and 0 <= xx < w and j in remaining:
                    remaining.remove(j)
                    pending.append(j)
                    found.append(j)
        groups.append(found)
    kept = max(groups, key=len) if groups else []
    output = bytearray(len(pixels))
    for i in kept:
        output[i] = 255
    view = grayscale_image(shape, bytes(output))
    original = sum(map(len, groups))
    return view, dict(
        version="component-measurement-view/1.0.0",
        method="largest_4_connected_foreground_no_fill_no_resize",
        original=checksum(raw),
        output=checksum(view),
        component_pixels=sorted(map(len, groups), reverse=True),
        retained_pixels=len(kept),
        excluded_pixels=original - len(kept),
        retained_fraction=len(kept) / original if original else 0.0,
        boundary_contact=any(i // w in (0, h - 1) or i % w in (0, w - 1) for i in kept),
        invented_pixels=0,
        clinical_accuracy=None,
        original_retained=True,
        independent_evidence=False,
    )
