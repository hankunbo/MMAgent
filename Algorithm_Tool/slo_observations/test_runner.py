"""CPU-only formula unit tests; no checkpoint loads or patient inputs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import runner


class Measurements(unittest.TestCase):
    def test_density(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[10, :] = True
        r = runner.measure(mask, "density")
        self.assertEqual(r["value"], 1 / 32)
        self.assertEqual(r["denominator"], 1024)

    def test_width(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[10, 3:29] = True
        r = runner.measure(mask, "width")
        self.assertEqual(r["value"], 1)
        self.assertEqual(r["unit"], "pixel")
        self.assertIsNone(r["physical_scale"])

    def test_straight_tortuosity(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[10, 3:29] = True
        r = runner.measure(mask, "tortuosity")
        self.assertEqual(r["value"], 1)
        self.assertEqual(len(r["segments"]), 1)

    def test_bent_tortuosity(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10, 5:25] = True
        mask[10:30, 24] = True
        r = runner.measure(mask, "tortuosity")
        self.assertGreater(r["value"], 1.3)
        self.assertEqual(len(r["segments"]), 1)

    def test_branching(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 5:60] = True
        mask[5:33, 32] = True
        r = runner.measure(mask, "tortuosity")
        self.assertEqual(len(r["segments"]), 3)
        self.assertEqual(r["value"], 1)

    def test_loop_excluded(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[5, 5:26] = mask[25, 5:26] = True
        mask[5:26, 5] = mask[5:26, 25] = True
        r = runner.measure(mask, "tortuosity")
        self.assertIsNone(r["value"])
        self.assertEqual(r["excluded"]["closed_or_zero_chord"], 1)

    def test_short_excluded(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[10, 3:7] = True
        self.assertIsNone(runner.measure(mask, "tortuosity")["value"])

    def test_fractal_line(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, :] = True
        r = runner.measure(mask, "fractal")
        self.assertAlmostEqual(r["value"], 1)
        self.assertEqual(r["box_sizes"], [64, 32, 16, 8, 4])
        self.assertEqual(r["box_counts"], [1, 2, 4, 8, 16])

    def test_fractal_insufficient(self):
        mask = np.ones((8, 8), dtype=bool)
        self.assertIsNone(runner.measure(mask, "fractal")["value"])

    def test_empty_all_metrics(self):
        for task in runner.METRICS:
            r = runner.measure(np.zeros((32, 32), dtype=bool), task)
            self.assertIsNone(r["value"])
            self.assertTrue(r["unavailable_reason"])
            json.dumps(r, allow_nan=False)

    def test_invalid_arrays(self):
        for arr in (np.zeros((32, 32)), np.zeros((2, 3, 4), dtype=bool)):
            with self.assertRaises(ValueError):
                runner.measure(arr, "density")

    def test_image_rejections_and_no_write(self):
        with tempfile.TemporaryDirectory(dir="/opt/data/test") as temp:
            root = Path(temp)
            rgb, grey, bad = root / "rgb.png", root / "grey.png", root / "bad.png"
            Image.new("RGB", (12, 12)).save(rgb)
            Image.new("L", (12, 12), color=120).save(grey)
            bad.write_bytes(b"not an image")
            for path in (rgb, grey, bad):
                with self.assertRaises((ValueError, OSError)):
                    runner.read_image(path, mask=True)
            output = root / "result.json"
            with patch.object(runner, "verify_installation", return_value="fixture"):
                with self.assertRaises(ValueError):
                    runner.main(["density", "--input", str(grey), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_deterministic_and_transpose(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 5:40] = True
        for task in runner.METRICS:
            a, b = runner.measure(mask, task), runner.measure(mask, task)
            self.assertEqual(a, b)
            self.assertAlmostEqual(a["value"], runner.measure(mask.T, task)["value"])

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory(dir="/opt/data/test") as temp:
            p = Path(temp) / "result.json"
            p.write_text("keep")
            with self.assertRaises(ValueError):
                runner.main(["density", "--input", "unused", "--output", str(p)])
            self.assertEqual(p.read_text(), "keep")

    def test_output_ports_rejected(self):
        with self.assertRaises(ValueError):
            runner.main(["vessels", "--input", "unused", "--output", "/opt/data/test/unused"])

    def test_installation_drift(self):
        with tempfile.TemporaryDirectory(dir="/opt/data/test") as temp:
            p, lock = Path(temp) / "source", Path(temp) / "lock.json"
            p.write_bytes(b"a")
            lock.write_text(
                json.dumps({"resources": [{"path": str(p), "bytes": 1, "sha256": runner.sha(p)}]})
            )
            with patch.object(runner, "LOCK", lock):
                runner.verify_installation()
                p.write_bytes(b"b")
                with self.assertRaisesRegex(ValueError, "drift"):
                    runner.verify_installation()


if __name__ == "__main__":
    unittest.main()
