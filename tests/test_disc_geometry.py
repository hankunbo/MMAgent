"""Analytic/negative fixtures, deterministic geometry and source-byte binding."""

import json
import math
import subprocess
import sys

import pytest

from apps import disc_geometry as g
from packages.dataset_adapters.reading import grayscale_image


def mask(a=20, b=20, angle=0, cx=45, cy=45, shape=(91, 91)):
    angle = math.radians(angle)
    pixels = bytearray(math.prod(shape))
    for y in range(shape[0]):
        for x in range(shape[1]):
            dx, dy = x - cx, y - cy
            xx = dx * math.cos(angle) + dy * math.sin(angle)
            yy = -dx * math.sin(angle) + dy * math.cos(angle)
            pixels[y * shape[1] + x] = 255 if xx * xx / a**2 + yy * yy / b**2 <= 1 else 0
    return pixels


@pytest.mark.parametrize("angle", [0, 30, 60, 90, 135])
def test_ellipse_axes_and_rotation(angle):
    result = g.ellipse((91, 91), mask(25, 12, angle))
    assert result["major_axis_px"] == pytest.approx(50, abs=0.6)
    assert result["minor_axis_px"] == pytest.approx(24, abs=0.6)
    assert result["minor_over_major"] == pytest.approx(12 / 25, abs=0.012)
    assert result["minor_over_major"] * result["major_over_minor"] == pytest.approx(1)
    assert abs((result["major_axis_angle_degrees"] - angle + 90) % 180 - 90) < 0.6


def test_circle_no_invented_angle():
    result = g.ellipse((91, 91), mask())
    assert result["minor_over_major"] == 1
    assert result["major_axis_angle_degrees"] is None
    assert result["angle_missing_reason"] == "near_circular_axis_not_identifiable"


def test_translation_preserves_ellipse_and_rim():
    a = g.ellipse((91, 91), mask(25, 12))
    b = g.ellipse((91, 91), mask(25, 12, cx=47, cy=39))
    assert a["major_axis_px"] == b["major_axis_px"]
    assert a["minor_axis_px"] == b["minor_axis_px"]
    a = g.rim((91, 91), mask(), mask(10, 10))
    b = g.rim((91, 91), mask(cx=47, cy=39), mask(10, 10, cx=47, cy=39))
    assert a["rays"] == b["rays"]


def test_rim_formula_full_distribution_and_unknown_anatomy():
    result = g.rim((91, 91), mask(), mask(10, 10))
    assert len(result["rays"]) == 360
    assert result["valid_rays"] + result["invalid_rays"] == 360
    for a in (0, 90, 180, 270):
        row = result["rays"][a]
        assert row["rim_width_px"] == 10
        # Digital circle has unit-square pixels; quarter-pixel first-exit is bounded.
        assert 41 <= row["disc_diameter_px"] <= 41.5
        assert row["width_over_same_axis_disc_diameter"] == 10 / row["disc_diameter_px"]
    assert sum(s["valid_rays"] for s in result["image_sectors"].values()) == result["valid_rays"]
    assert result["anatomical_comparison"] is None


@pytest.mark.parametrize("orientation", g.ORIENTATIONS[1:])
def test_explicit_cardinal_mapping_uses_raw_widths(orientation):
    result = g.rim(
        (91, 91), mask(), mask(10, 10, cx=47, cy=43), orientation, "synthetic_orientation_fixture"
    )
    compare = result["anatomical_comparison"]
    widths = compare["cardinal_widths_px"]
    assert widths["I"] == result["rays"][90]["rim_width_px"]
    nasal = 180 if orientation.endswith("left") else 0
    assert widths["N"] == result["rays"][nasal]["rim_width_px"]
    assert compare["ISNT_nonincreasing"] == (
        widths["I"] >= widths["S"] >= widths["N"] >= widths["T"]
    )


@pytest.mark.parametrize(
    "orientation,source",
    [
        ("unknown", None),
        ("superior_up_nasal_left", None),
        ("superior_up_nasal_left", ""),
        ("image_only", "cannot_use_fovea_guess"),
    ],
)
def test_bad_orientation(orientation, source):
    with pytest.raises(ValueError, match="orientation_definition_invalid"):
        g.rim((91, 91), mask(), mask(10, 10), orientation, source)


@pytest.mark.parametrize(
    "kind,code",
    [
        ("empty", "mask_empty"),
        ("labels", "mask_binary"),
        ("short", "mask_binary"),
        ("island", "mask_multiple"),
        ("hole", "mask_has_holes"),
        ("edge", "mask_clipped"),
        ("line", "mask_degenerate"),
    ],
)
def test_invalid_masks(kind, code):
    pixels = mask()
    if kind == "empty":
        pixels = bytes(len(pixels))
    elif kind == "labels":
        pixels[45 * 91 + 45] = 1
    elif kind == "short":
        pixels.pop()
    elif kind == "island":
        pixels[5 * 91 + 5] = 255
    elif kind == "hole":
        pixels[45 * 91 + 45] = 0
    elif kind == "edge":
        pixels = mask(cx=19)
    else:
        pixels = bytearray(91 * 91)
        pixels[45 * 91 + 40 : 45 * 91 + 46] = b"\xff" * 6
    with pytest.raises(ValueError, match=code):
        g.ellipse((91, 91), pixels)


def test_cup_containment_and_center():
    with pytest.raises(ValueError, match="cup_outside_disc"):
        g.rim((91, 91), mask(), mask(22, 22))
    with pytest.raises(ValueError, match="disc_centroid_outside_cup"):
        g.rim((91, 91), mask(), mask(4, 4, cx=55))


def test_pixel_scale_and_normalized_shape():
    a = g.ellipse((91, 91), mask(12, 6))
    b = g.ellipse((91, 91), mask(24, 12))
    assert b["major_axis_px"] / a["major_axis_px"] == pytest.approx(2, abs=0.05)
    assert a["minor_over_major"] == pytest.approx(b["minor_over_major"], abs=0.025)


def test_nonstar_boundary_is_explicit_not_silently_first_exit():
    # Connected, no holes, but a side notch makes some rays leave and re-enter.
    disc = mask(30, 30)
    for y in range(35, 40):
        for x in range(55, 76):
            disc[y * 91 + x] = 0
    result = g.rim((91, 91), disc, mask(8, 8))
    assert result["invalid_rays"] > 0
    assert all(r["rim_width_px"] is None for r in result["rays"] if r["missing_reason"])


def test_measure_binding_repeated_bytes_and_bad_shape(tmp_path):
    disc, cup = tmp_path / "disc.png", tmp_path / "cup.png"
    disc.write_bytes(grayscale_image((91, 91), mask()))
    cup.write_bytes(grayscale_image((91, 91), mask(10, 10)))
    first = g.measure("rim", disc, cup)
    assert first == g.measure("rim", disc, cup)
    assert first["inputs"]["disc_mask"]["sha256"] in first["correlation_group"]
    assert first["disease_probability"] is None
    assert "NaN" not in json.dumps(first, allow_nan=False)
    cup.write_bytes(grayscale_image((71, 71), mask(10, 10, cx=35, cy=35, shape=(71, 71))))
    with pytest.raises(ValueError, match="mask_shape_mismatch"):
        g.measure("rim", disc, cup)


@pytest.mark.parametrize(
    "task,cup,code",
    [
        ("rim", None, "cup_mask_required"),
        ("ellipse", True, "ellipse_extra_inputs"),
        ("guess", None, "unknown_geometry_task"),
    ],
)
def test_ports(task, cup, code, tmp_path):
    disc = tmp_path / "disc.png"
    disc.write_bytes(grayscale_image((91, 91), mask()))
    with pytest.raises(ValueError, match=code):
        g.measure(task, disc, disc if cup else None)


def test_cli_does_not_overwrite_or_write_on_invalid_input(tmp_path):
    disc, output = tmp_path / "disc.png", tmp_path / "result.json"
    disc.write_bytes(grayscale_image((91, 91), mask(20, 10)))
    command = [
        sys.executable,
        "-m",
        "apps.disc_geometry",
        "ellipse",
        "--disc-mask",
        str(disc),
        "--output",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr
    first = output.read_bytes()
    assert subprocess.run(command, capture_output=True, timeout=20).returncode != 0
    assert output.read_bytes() == first
    other = tmp_path / "must-not-exist.json"
    disc.write_bytes(grayscale_image((91, 91), bytes(91 * 91)))
    command[-1] = str(other)
    assert subprocess.run(command, capture_output=True, timeout=20).returncode != 0
    assert not other.exists()


def test_output_commit_failure_cleanup_and_nonfinite_rejection(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    with pytest.raises(ValueError):
        g.write_result(output, {"invalid": float("nan")})
    assert not output.exists()

    def fail(*args):
        raise OSError("synthetic_commit_failure")

    monkeypatch.setattr(g.os, "link", fail)
    with pytest.raises(OSError, match="synthetic_commit_failure"):
        g.write_result(output, {"complete": True})
    assert list(tmp_path.iterdir()) == []
