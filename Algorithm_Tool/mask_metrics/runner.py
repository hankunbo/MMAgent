#!/usr/bin/env python3
"""Deterministic morphometry tools for optic-disc, optic-cup, and vessel masks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


REFERENCE_URLS = {
    "optic_nerve_morphometry": "https://doi.org/10.3390/diagnostics12051063",
    "rim_area": "https://doi.org/10.1007/BF00164670",
    "vascx": "https://arxiv.org/abs/2602.08580",
    "pvbm": "https://arxiv.org/abs/2208.00392",
    "vessel_density_and_vdi": "https://doi.org/10.1117/1.JBO.21.6.066008",
    "edt_caliber": "https://doi.org/10.1049/iet-ipr.2020.0336",
}


def _read_mask(path: Path, name: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read {name}: {path}")
    return mask > 0


def _require_nonempty(mask: np.ndarray, name: str) -> None:
    if not mask.any():
        raise ValueError(f"{name} is empty")


def _require_same_shape(named_masks: dict[str, np.ndarray]) -> None:
    shapes = {name: mask.shape for name, mask in named_masks.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Mask shapes differ: {shapes}")


def _validate_disc_cup(disc: np.ndarray, cup: np.ndarray) -> dict[str, int]:
    _require_same_shape({"disc_mask": disc, "cup_mask": cup})
    _require_nonempty(disc, "disc_mask")
    _require_nonempty(cup, "cup_mask")
    outside = int(np.logical_and(cup, ~disc).sum())
    if outside:
        raise ValueError(
            f"The optic-cup mask has {outside} foreground pixels outside the disc mask"
        )
    return {
        "disc_area_px": int(disc.sum()),
        "cup_area_px": int(cup.sum()),
        "rim_area_px": int(np.logical_and(disc, ~cup).sum()),
        "cup_pixels_outside_disc": outside,
    }


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _base_result(metric: str, formula: str, unit: str, sources: Iterable[str]) -> dict[str, object]:
    return {
        "metric": metric,
        "formula": formula,
        "unit": unit,
        "implementation": "MMAgent mask_metrics 1.0.0",
        "sources": [REFERENCE_URLS[key] for key in sources],
        "warnings": [],
    }


def calculate_hcdr(disc: np.ndarray, cup: np.ndarray) -> dict[str, object]:
    diagnostics = _validate_disc_cup(disc, cup)
    _, _, disc_left, disc_right = _bbox(disc)
    _, _, cup_left, cup_right = _bbox(cup)
    disc_width = disc_right - disc_left + 1
    cup_width = cup_right - cup_left + 1
    result = _base_result(
        "hcdr",
        "cup_horizontal_span_px / disc_horizontal_span_px",
        "1",
        ("optic_nerve_morphometry",),
    )
    result.update(
        value=float(cup_width / disc_width),
        disc_horizontal_span_px=disc_width,
        cup_horizontal_span_px=cup_width,
        diagnostics=diagnostics,
    )
    return result


def calculate_rdar(disc: np.ndarray, cup: np.ndarray) -> dict[str, object]:
    diagnostics = _validate_disc_cup(disc, cup)
    value = diagnostics["rim_area_px"] / diagnostics["disc_area_px"]
    result = _base_result(
        "rdar",
        "(disc_area_px - cup_area_px) / disc_area_px",
        "1",
        ("optic_nerve_morphometry", "rim_area"),
    )
    result.update(value=float(value), diagnostics=diagnostics)
    return result


def calculate_rim_profile(disc: np.ndarray, cup: np.ndarray) -> dict[str, object]:
    diagnostics = _validate_disc_cup(disc, cup)
    disc_top, disc_bottom, disc_left, disc_right = _bbox(disc)
    cup_top, cup_bottom, cup_left, cup_right = _bbox(cup)
    disc_height = disc_bottom - disc_top + 1
    disc_width = disc_right - disc_left + 1
    widths = {
        "superior_px": int(cup_top - disc_top),
        "inferior_px": int(disc_bottom - cup_bottom),
        "left_px": int(cup_left - disc_left),
        "right_px": int(disc_right - cup_right),
    }
    if min(widths.values()) < 0:
        raise ValueError("Cup bounding box extends beyond the disc bounding box")
    ratios = {
        "superior_to_disc_height": widths["superior_px"] / disc_height,
        "inferior_to_disc_height": widths["inferior_px"] / disc_height,
        "left_to_disc_width": widths["left_px"] / disc_width,
        "right_to_disc_width": widths["right_px"] / disc_width,
    }
    result = _base_result(
        "rim_width_profile",
        "axis_aligned_disc_boundary_minus_cup_boundary",
        "pixel",
        ("optic_nerve_morphometry",),
    )
    result.update(
        values=widths,
        normalized_values={key: float(value) for key, value in ratios.items()},
        disc_bbox_ymin_ymax_xmin_xmax=[disc_top, disc_bottom, disc_left, disc_right],
        cup_bbox_ymin_ymax_xmin_xmax=[cup_top, cup_bottom, cup_left, cup_right],
        diagnostics=diagnostics,
    )
    result["warnings"].append(
        "Left/right are image coordinates; nasal/temporal and the ISNT rule require eye laterality."
    )
    result["warnings"].append(
        "These are axis-aligned bounding-box rim widths, not OCT BMO-MRW measurements."
    )
    return result


def _skeleton_data(vessel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _require_nonempty(vessel, "vessel_mask")
    skeleton = skeletonize(vessel)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    return skeleton, neighbors


def _graph_length(skeleton: np.ndarray) -> float:
    horizontal = np.logical_and(skeleton[:, :-1], skeleton[:, 1:]).sum()
    vertical = np.logical_and(skeleton[:-1, :], skeleton[1:, :]).sum()
    diagonal_1 = np.logical_and(skeleton[:-1, :-1], skeleton[1:, 1:]).sum()
    diagonal_2 = np.logical_and(skeleton[:-1, 1:], skeleton[1:, :-1]).sum()
    return float(horizontal + vertical + math.sqrt(2.0) * (diagonal_1 + diagonal_2))


def _quality_warnings(vessel: np.ndarray) -> list[str]:
    warnings: list[str] = []
    fraction = float(vessel.mean())
    components = int(
        ndimage.label(vessel, structure=np.ones((3, 3), dtype=np.uint8))[1]
    )
    if fraction < 0.01:
        warnings.append(
            "Vessel foreground is below 1% of the image; downstream morphometry may reflect an incomplete vessel tree."
        )
    if components > 50:
        warnings.append(
            "The vessel mask is highly fragmented; endpoints, junctions, and tortuosity are sensitive to fragmentation."
        )
    return warnings


def calculate_vessel_density(vessel: np.ndarray) -> dict[str, object]:
    _require_nonempty(vessel, "vessel_mask")
    vessel_pixels = int(vessel.sum())
    frame_pixels = int(vessel.size)
    result = _base_result(
        "vessel_area_density_frame",
        "vessel_pixels / image_frame_pixels",
        "1",
        ("vascx", "vessel_density_and_vdi"),
    )
    result.update(
        value=float(vessel_pixels / frame_pixels),
        vessel_pixels=vessel_pixels,
        roi_pixels=frame_pixels,
        roi="full_image_frame",
    )
    result["warnings"].extend(_quality_warnings(vessel))
    result["warnings"].append(
        "The denominator is the full image frame; use a retinal/FOV ROI for cross-device comparisons."
    )
    return result


def calculate_vessel_topology(vessel: np.ndarray) -> dict[str, object]:
    skeleton, neighbors = _skeleton_data(vessel)
    skeleton_pixels = int(skeleton.sum())
    graph_length = _graph_length(skeleton)
    endpoints = int(np.logical_and(skeleton, neighbors == 1).sum())
    branch_pixels = np.logical_and(skeleton, neighbors >= 3)
    branch_clusters = int(
        ndimage.label(branch_pixels, structure=np.ones((3, 3), dtype=np.uint8))[1]
    )
    components = int(
        ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))[1]
    )
    result = _base_result(
        "vessel_topology_summary",
        "8_connected_skeleton_graph",
        "mixed",
        ("vascx", "pvbm"),
    )
    result.update(
        values={
            "skeleton_pixels": skeleton_pixels,
            "skeleton_graph_length_px": graph_length,
            "skeleton_pixel_density": float(skeleton_pixels / vessel.size),
            "skeleton_length_density_px_per_pixel2": float(graph_length / vessel.size),
            "endpoint_candidates": endpoints,
            "branchpoint_candidate_clusters": branch_clusters,
            "connected_components": components,
        }
    )
    result["warnings"].extend(_quality_warnings(vessel))
    result["warnings"].append(
        "Branchpoint candidates cannot distinguish true bifurcations from vessel crossings in a binary mask."
    )
    return result


def calculate_vessel_caliber(
    vessel: np.ndarray, mm_per_pixel: float | None
) -> dict[str, object]:
    skeleton, _ = _skeleton_data(vessel)
    skeleton_pixels = int(skeleton.sum())
    if skeleton_pixels == 0:
        raise ValueError("The skeleton is empty")
    distance = ndimage.distance_transform_edt(vessel)
    diameters = 2.0 * distance[skeleton]
    vessel_area = int(vessel.sum())
    vdi = vessel_area / skeleton_pixels
    values: dict[str, object] = {
        "vessel_diameter_index_px": float(vdi),
        "edt_diameter_median_px": float(np.median(diameters)),
        "edt_diameter_mean_px": float(np.mean(diameters)),
        "edt_diameter_p25_px": float(np.percentile(diameters, 25)),
        "edt_diameter_p75_px": float(np.percentile(diameters, 75)),
        "diameter_sample_count": int(diameters.size),
    }
    unit = "pixel"
    if mm_per_pixel is not None:
        if mm_per_pixel <= 0:
            raise ValueError("--mm-per-pixel must be positive")
        values.update(
            vessel_diameter_index_mm=float(vdi * mm_per_pixel),
            edt_diameter_median_mm=float(np.median(diameters) * mm_per_pixel),
            edt_diameter_mean_mm=float(np.mean(diameters) * mm_per_pixel),
        )
        unit = "pixel_and_mm"
    result = _base_result(
        "vessel_caliber_summary",
        "VDI=vessel_area_px/skeleton_pixels; EDT_diameter=2*distance_to_background_at_skeleton",
        unit,
        ("vessel_density_and_vdi", "edt_caliber"),
    )
    result.update(values=values, mm_per_pixel=mm_per_pixel)
    result["warnings"].extend(_quality_warnings(vessel))
    result["warnings"].append(
        "EDT diameter is a reproducible mask-derived approximation and is not CRAE/CRVE."
    )
    return result


_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)


def _pixel_neighbors(point: tuple[int, int], points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    y, x = point
    return [(y + dy, x + dx) for dy, dx in _NEIGHBOR_OFFSETS if (y + dy, x + dx) in points]


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _trace_skeleton_segments(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    points = {tuple(map(int, point)) for point in np.argwhere(skeleton)}
    adjacency = {point: _pixel_neighbors(point, points) for point in points}
    nodes = {point for point, neighbors in adjacency.items() if len(neighbors) != 2}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []

    def trace(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, first]
        visited.add(_edge_key(start, first))
        previous, current = start, first
        while current not in nodes:
            candidates = [point for point in adjacency[current] if point != previous]
            if not candidates:
                break
            next_point = candidates[0]
            edge = _edge_key(current, next_point)
            if edge in visited:
                break
            visited.add(edge)
            path.append(next_point)
            previous, current = current, next_point
        return path

    for start in sorted(nodes):
        for neighbor in adjacency[start]:
            edge = _edge_key(start, neighbor)
            if edge not in visited:
                segments.append(trace(start, neighbor))

    for start in sorted(points):
        for neighbor in adjacency[start]:
            edge = _edge_key(start, neighbor)
            if edge not in visited:
                segments.append(trace(start, neighbor))
    return segments


def _path_lengths(path: list[tuple[int, int]]) -> tuple[float, float]:
    coordinates = np.asarray(path, dtype=np.float64)
    differences = np.diff(coordinates, axis=0)
    arc = float(np.sqrt(np.square(differences).sum(axis=1)).sum())
    chord = float(np.linalg.norm(coordinates[-1] - coordinates[0]))
    return arc, chord


def calculate_vessel_tortuosity(
    vessel: np.ndarray, minimum_segment_length: float
) -> dict[str, object]:
    if minimum_segment_length <= 0:
        raise ValueError("--min-segment-length must be positive")
    skeleton, _ = _skeleton_data(vessel)
    segments = _trace_skeleton_segments(skeleton)
    eligible: list[tuple[float, float]] = []
    for segment in segments:
        arc, chord = _path_lengths(segment)
        if arc >= minimum_segment_length and chord > 0:
            eligible.append((arc / chord, arc))
    if not eligible:
        raise ValueError("No eligible vessel segment for tortuosity")
    factors = np.asarray([item[0] for item in eligible], dtype=np.float64)
    lengths = np.asarray([item[1] for item in eligible], dtype=np.float64)
    result = _base_result(
        "vessel_distance_tortuosity_summary",
        "segment_arc_length / segment_endpoint_chord_length",
        "1",
        ("vascx", "pvbm"),
    )
    result.update(
        values={
            "median_distance_factor": float(np.median(factors)),
            "mean_distance_factor": float(np.mean(factors)),
            "length_weighted_distance_factor": float(np.average(factors, weights=lengths)),
            "eligible_segment_count": int(factors.size),
            "traced_segment_count": int(len(segments)),
            "minimum_segment_length_px": float(minimum_segment_length),
        }
    )
    result["warnings"].extend(_quality_warnings(vessel))
    return result


def calculate_vessel_fractal_dimension(vessel: np.ndarray) -> dict[str, object]:
    skeleton, _ = _skeleton_data(vessel)
    height, width = skeleton.shape
    maximum_power = int(math.floor(math.log2(min(height, width))))
    sizes = [2 ** power for power in range(1, maximum_power) if 2 ** power <= min(height, width) // 2]
    ys, xs = np.nonzero(skeleton)
    counts: list[int] = []
    used_sizes: list[int] = []
    for size in sizes:
        boxes = np.unique(np.column_stack((ys // size, xs // size)), axis=0)
        count = int(len(boxes))
        if count > 1:
            used_sizes.append(size)
            counts.append(count)
    if len(counts) < 3:
        raise ValueError("At least three valid box sizes are required for fractal dimension")
    x = np.log(1.0 / np.asarray(used_sizes, dtype=np.float64))
    y = np.log(np.asarray(counts, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.square(y - predicted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    result = _base_result(
        "vessel_skeleton_box_counting_dimension",
        "slope(log(N(box_size)), log(1/box_size))",
        "1",
        ("pvbm",),
    )
    result.update(
        value=float(slope),
        box_sizes_px=used_sizes,
        occupied_box_counts=counts,
        regression_r_squared=float(r_squared),
    )
    result["warnings"].extend(_quality_warnings(vessel))
    return result


def calculate_vessel_sparsity(vessel: np.ndarray) -> dict[str, object]:
    _require_nonempty(vessel, "vessel_mask")
    distance = ndimage.distance_transform_edt(~vessel)
    values = distance.reshape(-1)
    result = _base_result(
        "vessel_sparsity_frame",
        "distance_transform_to_nearest_vessel_over_image_frame",
        "pixel",
        ("vascx",),
    )
    result.update(
        values={
            "mean_nearest_vessel_distance_px": float(np.mean(values)),
            "p95_nearest_vessel_distance_px": float(np.percentile(values, 95)),
            "maximum_nearest_vessel_distance_px": float(np.max(values)),
        },
        roi="full_image_frame",
    )
    result["warnings"].extend(_quality_warnings(vessel))
    result["warnings"].append(
        "The full frame includes non-retinal background unless an upstream FOV mask has already cropped it."
    )
    return result


def calculate_regional_density(
    region: str,
    vessel: np.ndarray,
    disc: np.ndarray | None,
    cup: np.ndarray | None,
) -> dict[str, object]:
    masks = {"vessel_mask": vessel}
    if disc is not None:
        masks["disc_mask"] = disc
    if cup is not None:
        masks["cup_mask"] = cup
    _require_same_shape(masks)
    _require_nonempty(vessel, "vessel_mask")

    if region == "disc":
        assert disc is not None
        _require_nonempty(disc, "disc_mask")
        roi = disc
    elif region == "cup":
        assert cup is not None
        _require_nonempty(cup, "cup_mask")
        roi = cup
    else:
        assert disc is not None and cup is not None
        _validate_disc_cup(disc, cup)
        roi = np.logical_and(disc, ~cup)

    roi_pixels = int(roi.sum())
    if roi_pixels == 0:
        raise ValueError(f"The {region} ROI is empty")
    vessel_pixels = int(np.logical_and(vessel, roi).sum())
    result = _base_result(
        f"vessel_area_density_{region}",
        "count(vessel_mask AND roi_mask) / count(roi_mask)",
        "1",
        ("vascx",),
    )
    result.update(
        value=float(vessel_pixels / roi_pixels),
        vessel_pixels_in_roi=vessel_pixels,
        roi_pixels=roi_pixels,
        roi=region,
    )
    result["warnings"].extend(_quality_warnings(vessel))
    result["warnings"].append(
        "This is color-fundus vessel-mask occupancy, not OCTA perfusion density."
    )
    return result


METRICS = (
    "hcdr",
    "rdar",
    "rim-profile",
    "vessel-density",
    "vessel-topology",
    "vessel-caliber",
    "vessel-tortuosity",
    "vessel-fractal-dimension",
    "vessel-sparsity",
    "disc-vessel-density",
    "cup-vessel-density",
    "rim-vessel-density",
)


def calculate(args: argparse.Namespace) -> dict[str, object]:
    disc = _read_mask(args.disc_mask, "disc_mask") if args.disc_mask else None
    cup = _read_mask(args.cup_mask, "cup_mask") if args.cup_mask else None
    vessel = _read_mask(args.vessel_mask, "vessel_mask") if args.vessel_mask else None

    if args.metric in {"hcdr", "rdar", "rim-profile"}:
        if disc is None or cup is None:
            raise ValueError(f"{args.metric} requires --disc-mask and --cup-mask")
        result = {
            "hcdr": calculate_hcdr,
            "rdar": calculate_rdar,
            "rim-profile": calculate_rim_profile,
        }[args.metric](disc, cup)
    elif args.metric == "vessel-density":
        if vessel is None:
            raise ValueError("vessel-density requires --vessel-mask")
        result = calculate_vessel_density(vessel)
    elif args.metric == "vessel-topology":
        if vessel is None:
            raise ValueError("vessel-topology requires --vessel-mask")
        result = calculate_vessel_topology(vessel)
    elif args.metric == "vessel-caliber":
        if vessel is None:
            raise ValueError("vessel-caliber requires --vessel-mask")
        result = calculate_vessel_caliber(vessel, args.mm_per_pixel)
    elif args.metric == "vessel-tortuosity":
        if vessel is None:
            raise ValueError("vessel-tortuosity requires --vessel-mask")
        result = calculate_vessel_tortuosity(vessel, args.min_segment_length)
    elif args.metric == "vessel-fractal-dimension":
        if vessel is None:
            raise ValueError("vessel-fractal-dimension requires --vessel-mask")
        result = calculate_vessel_fractal_dimension(vessel)
    elif args.metric == "vessel-sparsity":
        if vessel is None:
            raise ValueError("vessel-sparsity requires --vessel-mask")
        result = calculate_vessel_sparsity(vessel)
    else:
        if vessel is None:
            raise ValueError(f"{args.metric} requires --vessel-mask")
        region = args.metric.removesuffix("-vessel-density")
        if region == "disc" and disc is None:
            raise ValueError("disc-vessel-density requires --disc-mask")
        if region == "cup" and cup is None:
            raise ValueError("cup-vessel-density requires --cup-mask")
        if region == "rim" and (disc is None or cup is None):
            raise ValueError("rim-vessel-density requires --disc-mask and --cup-mask")
        result = calculate_regional_density(region, vessel, disc, cup)

    input_uris: dict[str, str] = {}
    for name in ("disc_mask", "cup_mask", "vessel_mask"):
        path = getattr(args, name)
        if path is not None:
            input_uris[name] = path.resolve().as_uri()
    result["input_uris"] = input_uris
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=METRICS)
    parser.add_argument("--disc-mask", type=Path)
    parser.add_argument("--cup-mask", type=Path)
    parser.add_argument("--vessel-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mm-per-pixel", type=float)
    parser.add_argument("--min-segment-length", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calculate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
