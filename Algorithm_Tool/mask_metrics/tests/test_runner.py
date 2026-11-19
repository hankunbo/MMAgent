#!/usr/bin/env python3
"""Unit tests for deterministic mask-derived formulas."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


RUNNER_PATH = Path(__file__).resolve().parents[1] / "runner.py"
SPEC = importlib.util.spec_from_file_location("mask_metrics_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OpticNerveMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.disc = np.zeros((100, 100), dtype=bool)
        self.cup = np.zeros_like(self.disc)
        self.disc[20:80, 10:90] = True
        self.cup[30:70, 30:70] = True

    def test_hcdr(self) -> None:
        result = runner.calculate_hcdr(self.disc, self.cup)
        self.assertAlmostEqual(result["value"], 0.5)
        self.assertEqual(result["disc_horizontal_span_px"], 80)
        self.assertEqual(result["cup_horizontal_span_px"], 40)

    def test_rdar(self) -> None:
        result = runner.calculate_rdar(self.disc, self.cup)
        self.assertAlmostEqual(result["value"], 2.0 / 3.0)
        self.assertEqual(result["diagnostics"]["rim_area_px"], 3200)

    def test_rim_profile(self) -> None:
        result = runner.calculate_rim_profile(self.disc, self.cup)
        self.assertEqual(
            result["values"],
            {"superior_px": 10, "inferior_px": 10, "left_px": 20, "right_px": 20},
        )

    def test_rejects_cup_outside_disc(self) -> None:
        invalid = self.cup.copy()
        invalid[0, 0] = True
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.calculate_rdar(self.disc, invalid)


class VesselMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vessel = np.zeros((64, 64), dtype=bool)
        self.vessel[30:33, 10:54] = True

    def test_density(self) -> None:
        result = runner.calculate_vessel_density(self.vessel)
        self.assertAlmostEqual(result["value"], 132 / 4096)

    def test_topology_straight_segment(self) -> None:
        result = runner.calculate_vessel_topology(self.vessel)
        values = result["values"]
        self.assertEqual(values["endpoint_candidates"], 2)
        self.assertEqual(values["branchpoint_candidate_clusters"], 0)
        self.assertEqual(values["connected_components"], 1)

    def test_tortuosity_straight_segment(self) -> None:
        thin_vessel = np.zeros_like(self.vessel)
        thin_vessel[31, 10:54] = True
        result = runner.calculate_vessel_tortuosity(thin_vessel, 5.0)
        self.assertAlmostEqual(result["values"]["median_distance_factor"], 1.0)

    def test_caliber_and_vdi(self) -> None:
        result = runner.calculate_vessel_caliber(self.vessel, None)
        values = result["values"]
        self.assertGreater(values["edt_diameter_median_px"], 0)
        self.assertGreater(values["vessel_diameter_index_px"], 0)

    def test_fractal_dimension_of_line_is_near_one(self) -> None:
        result = runner.calculate_vessel_fractal_dimension(self.vessel)
        self.assertGreater(result["value"], 0.7)
        self.assertLess(result["value"], 1.3)

    def test_regional_density(self) -> None:
        disc = np.zeros_like(self.vessel)
        disc[20:40, 0:64] = True
        result = runner.calculate_regional_density(
            "disc", self.vessel, disc=disc, cup=None
        )
        self.assertAlmostEqual(result["value"], 132 / 1280)


if __name__ == "__main__":
    unittest.main()
