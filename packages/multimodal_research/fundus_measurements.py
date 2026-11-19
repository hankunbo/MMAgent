"""Reuse original mask/quality/geometry owners, independent of image modality."""

from pathlib import Path

from apps.disc_geometry import rim
from packages.evaluation_preparation.quality import mask_quality
from packages.tool_benchmark.geometry_metrics import ratios
from packages.tool_execution.qc import grayscale_png


def measure(directory):
    directory = Path(directory)
    raw = {n: (directory / (n + ".png")).read_bytes() for n in ("disc", "cup", "rim", "labels")}
    masks = {n: grayscale_png(v, 1048576) for n, v in raw.items()}
    shape, labels = masks["labels"]
    if (
        any(s != shape for s, _ in masks.values())
        or shape != (512, 512)
        or not set(labels) <= {0, 1, 2}
    ):
        raise ValueError("fundus_mask_shape_or_labels")
    for n, selected in (("disc", {1, 2}), ("cup", {2}), ("rim", {1})):
        if masks[n][1] != bytes(255 if v in selected else 0 for v in labels):
            raise ValueError("fundus_mask_class_binding")
    qc = mask_quality(raw["disc"], raw["cup"])
    values = ratios(labels, shape)
    for n in ("disc", "cup"):
        indices = [i for i, v in enumerate(masks[n][1]) if v]
        values[n + "_bbox_xyxy"] = (
            [
                min(i % 512 for i in indices),
                min(i // 512 for i in indices),
                max(i % 512 for i in indices) + 1,
                max(i // 512 for i in indices) + 1,
            ]
            if indices
            else None
        )
        values[n + "_area_px"] = len(indices)
    d, c = values["disc_bbox_xyxy"], values["cup_bbox_xyxy"]
    values["horizontal_cdr"] = (c[2] - c[0]) / (d[2] - d[0]) if d and c else None
    try:
        radial = rim(shape, masks["disc"][1], masks["cup"][1])
        radial_reason = None
    except ValueError as exc:
        radial, radial_reason = None, str(exc)
    return dict(
        schema_version="fundus-owned-geometry/1.0.0",
        status="obtained" if qc["measurement_available"] else "missing",
        values=values,
        radial_rim=radial,
        radial_missing_reason=radial_reason,
        qc=qc,
        quality=qc["quality_factor"],
        coordinate_system="512_pixel_analysis_grid",
        physical_scale=None,
        direction=None,
        quality_is_accuracy=False,
    )
