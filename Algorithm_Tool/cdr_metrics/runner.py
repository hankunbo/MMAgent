#!/usr/bin/env python3
"""Deterministic vCDR and aCDR calculators for typed optic-disc/cup masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask > 0


def _vertical_diameter(mask: np.ndarray) -> int:
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def calculate(metric: str, disc_path: Path, cup_path: Path) -> dict[str, object]:
    disc = _read_mask(disc_path)
    cup = _read_mask(cup_path)
    if disc.shape != cup.shape:
        raise ValueError(f"Mask shapes differ: disc={disc.shape}, cup={cup.shape}")

    disc_area = int(disc.sum())
    cup_area = int(cup.sum())
    outside_pixels = int(np.logical_and(cup, np.logical_not(disc)).sum())
    if disc_area == 0:
        raise ValueError("The optic-disc mask is empty")
    if outside_pixels:
        raise ValueError(
            f"The optic-cup mask has {outside_pixels} foreground pixels outside the disc mask"
        )

    disc_height = _vertical_diameter(disc)
    cup_height = _vertical_diameter(cup)
    if metric == "vcdr":
        if disc_height == 0:
            raise ValueError("The optic-disc vertical diameter is zero")
        value = cup_height / disc_height
        expression = "cup_vertical_diameter_px / disc_vertical_diameter_px"
    else:
        value = cup_area / disc_area
        expression = "cup_area_px / disc_area_px"

    return {
        "metric": metric,
        "value": float(value),
        "unit": "1",
        "expression": expression,
        "disc_vertical_diameter_px": disc_height,
        "cup_vertical_diameter_px": cup_height,
        "disc_area_px": disc_area,
        "cup_area_px": cup_area,
        "cup_pixels_outside_disc": outside_pixels,
        "disc_mask_uri": disc_path.resolve().as_uri(),
        "cup_mask_uri": cup_path.resolve().as_uri(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=("vcdr", "acdr"))
    parser.add_argument("--disc-mask", type=Path, required=True)
    parser.add_argument("--cup-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = calculate(args.metric, args.disc_mask, args.cup_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
